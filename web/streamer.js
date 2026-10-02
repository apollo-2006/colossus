// which pages live in the gpu pool: viewer/streamer.hpp ported, with http range
// requests for file reads. that file explains why these keep cuts whole:
//   * loads issue dependencies first and publish in issue order;
//   * a page depends on its dependencies from issue, and is evicted only when nothing
//     depends on it;
//   * pages below a request prefetch while their error, at the request's screen
//     scale, stays over the threshold.
const NONE = 0xffffffff;
const MAX_GAP = 64 << 10, MAX_RUN = 4 << 20;  // merging range requests: as viewer/streamer.hpp

export class Streamer {
  // pages: [{url, offset, size, deps: [global page], children: [global page], error, pinned}]
  constructor(pages, poolBytes, { maxInFlight = 256, uploadBytes = 16 << 20 } = {}) {
    this.pages = pages;
    let largest = 16;
    for (const p of pages) largest = Math.max(largest, p.size);
    this.slotBytes = Math.ceil(largest / 4096) * 4096;
    this.slotCount = Math.floor(poolBytes / this.slotBytes);
    if (this.slotCount < 2) throw new Error('page pool too small for one page');
    this.free = [];
    for (let s = this.slotCount - 1; s >= 0; s--) this.free.push(s);
    const n = pages.length;
    this.slotOf = new Uint32Array(n).fill(NONE);
    this.loading = new Uint8Array(n);
    this.lastUsed = new Uint32Array(n);
    this.dependents = new Uint32Array(n);
    this.table = new Uint32Array(n).fill(NONE);
    this.issued = [];  // {page, data: ArrayBuffer | null, failed}
    this.maxInFlight = maxInFlight;
    this.uploadBytes = uploadBytes;
    this.frame = 0;
    this.stats = { resident: 0, loaded: 0, inFlight: 0, bytes: 0, reads: 0 };
    this.batch = [];  // issued this frame, not yet fetched
    this.wholeFiles = new Map();  // url -> Promise<ArrayBuffer>, for servers ignoring ranges
  }

  resident(p) { return this.slotOf[p] !== NONE && !this.loading[p]; }

  noteUsed(stamps) {
    for (let p = 0; p < stamps.length; p++) if (stamps[p] > this.lastUsed[p]) this.lastUsed[p] = stamps[p];
  }

  // publishes finished loads, issues new ones. returns pool uploads ({slot, data});
  // this.table is then the frame's page table.
  service(frame, requests, threshold) {
    this.frame = frame;
    const uploads = [];
    let used = 0;
    while (this.issued.length && this.issued[0].data && used < this.uploadBytes) {
      const j = this.issued.shift();
      this.loading[j.page] = 0;
      this.table[j.page] = this.slotOf[j.page] * this.slotBytes / 4;
      uploads.push({ slot: this.slotOf[j.page], data: j.data });
      used += j.data.byteLength;
      this.stats.loaded++;
      this.stats.bytes += j.data.byteLength;
    }
    // prefetch below each request, then each page once at its best priority.
    const best = new Map();
    const want = (page, priority) => {
      if (page >= this.pages.length) return;
      if ((best.get(page) ?? -1) < priority) best.set(page, priority);
    };
    for (const [page, priority] of requests) want(page, priority);
    for (let r = 0; r < requests.length && best.size < 4096; r++) {
      const stack = [[requests[r][0], requests[r][1]]];
      while (stack.length && best.size < 4096) {
        const [page, priority] = stack.pop();
        const pg = this.pages[page];
        const known = pg.error > 0 && Number.isFinite(pg.error);
        for (const c of pg.children) {
          const predicted = known ? priority * (this.pages[c].error / pg.error) : priority * 0.5;
          if (predicted <= threshold) continue;
          want(c, predicted);
          stack.push([c, predicted]);
        }
      }
    }
    for (let p = 0; p < this.pages.length; p++) if (this.pages[p].pinned && this.slotOf[p] === NONE) want(p, Infinity);
    const order = [...best.entries()].sort((a, b) => b[1] - a[1]);
    this.candidates = null;
    for (const [page] of order) {
      if (this.slotOf[page] !== NONE) continue;
      const chain = [];
      this.collect(page, chain);
      for (const p of chain) if (this.issued.length >= this.maxInFlight || !this.issue(p)) break;
      if (this.issued.length >= this.maxInFlight) break;
    }
    this.flush();
    this.stats.inFlight = this.issued.length;
    this.stats.resident = this.slotCount - this.free.length - this.issued.length;
    return uploads;
  }

  collect(page, chain) {
    if (this.slotOf[page] !== NONE) {
      this.lastUsed[page] = this.frame;
      return;
    }
    if (chain.includes(page)) return;
    for (const d of this.pages[page].deps) this.collect(d, chain);
    chain.push(page);
  }

  evictable(p) {
    return this.resident(p) && !this.pages[p].pinned && this.dependents[p] === 0 && this.lastUsed[p] + 2 <= this.frame;
  }

  takeSlot() {
    if (this.free.length) return this.free.pop();
    if (!this.candidates) {
      this.candidates = [];
      for (let p = 0; p < this.pages.length; p++) if (this.evictable(p)) this.candidates.push(p);
      this.candidates.sort((a, b) => this.lastUsed[b] - this.lastUsed[a]);
    }
    while (this.candidates.length) {
      const p = this.candidates.pop();
      if (!this.evictable(p)) continue;
      const s = this.slotOf[p];
      this.slotOf[p] = NONE;
      this.table[p] = NONE;
      for (const d of this.pages[p].deps) this.dependents[d]--;
      return s;
    }
    return NONE;
  }

  // this frame's issued pages, sorted by place in the file and merged into range requests:
  // the file is laid out depth first, so a request and the pages prefetched below it are mostly
  // neighbours. gaps up to MAX_GAP are fetched and dropped.
  flush() {
    const pages = this.pages;
    this.batch.sort((a, b) => {
      const x = pages[a.page], y = pages[b.page];
      return x.url < y.url ? -1 : x.url > y.url ? 1 : x.offset - y.offset;
    });
    const runs = [];
    for (const job of this.batch) {
      const pg = pages[job.page];
      const r = runs[runs.length - 1];
      if (r && r.url === pg.url && pg.offset >= r.end && pg.offset - r.end <= MAX_GAP && pg.offset + pg.size - r.start <= MAX_RUN) {
        r.end = pg.offset + pg.size;
        r.jobs.push(job);
      } else {
        runs.push({ url: pg.url, start: pg.offset, end: pg.offset + pg.size, jobs: [job] });
      }
    }
    this.batch = [];
    this.stats.reads += runs.length;
    for (const r of runs) {
      this.fetchRange(r.url, r.start, r.end).then(
        (data) => {
          for (const job of r.jobs) {
            const pg = pages[job.page];
            job.data = data.slice(pg.offset - r.start, pg.offset - r.start + pg.size);
          }
        },
        (e) => console.error(`pages ${r.jobs.map((j) => j.page).join(', ')}:`, e),
      );
    }
  }

  async fetchRange(url, start, end) {
    for (let attempt = 0; ; attempt++) {
      try {
        const whole = this.wholeFiles.get(url);
        if (whole) return (await whole).slice(start, end);
        const r = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
        if (r.status === 206) return r.arrayBuffer();
        // a server ignoring ranges sends the whole file: keep it and take every page from it.
        if (r.ok) {
          const file = r.arrayBuffer();
          this.wholeFiles.set(url, file);
          return (await file).slice(start, end);
        }
        throw new Error(`HTTP ${r.status}`);
      } catch (e) {
        if (attempt >= 3) throw e;
        await new Promise((res) => setTimeout(res, 200 * (attempt + 1)));
      }
    }
  }

  issue(p) {
    const s = this.takeSlot();
    if (s === NONE) return false;
    const pg = this.pages[p];
    this.slotOf[p] = s;
    this.loading[p] = 1;
    this.lastUsed[p] = this.frame;
    for (const d of pg.deps) this.dependents[d]++;
    const job = { page: p, data: null };
    this.issued.push(job);
    this.batch.push(job);  // fetched by flush()
    return true;
  }
}
