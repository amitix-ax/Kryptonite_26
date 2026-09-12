/* Timeline Playback Controller — Antarctic Iceberg DSS */

class TimelineController {
  constructor(onStepChange) {
    this.onStepChange = onStepChange;
    this.steps = [];
    this.idx = 0;
    this.playing = false;
    this.speed = 1;
    this.timer = null;

    this.scrubber = document.getElementById('timeScrubber');
    this.btnPP = document.getElementById('btnPlayPause');
    this.btnPrev = document.getElementById('btnPrev');
    this.btnNext = document.getElementById('btnNext');
    this.timeUTC = document.getElementById('timeUTC');
    this.timeStep = document.getElementById('timeStep');

    this._bind();
  }

  init(steps) {
    this.steps = steps || [];
    this.idx = 0;
    this.scrubber.max = Math.max(this.steps.length - 1, 0);
    this.scrubber.value = 0;
    this._emit();
  }

  _bind() {
    this.scrubber.addEventListener('input', e => {
      this.idx = parseInt(e.target.value, 10);
      this._emit();
    });

    this.btnPP.addEventListener('click', () => this.playing ? this.pause() : this.play());
    this.btnPrev.addEventListener('click', () => { this.pause(); this._step(-1); });
    this.btnNext.addEventListener('click', () => { this.pause(); this._step(1); });

    document.querySelectorAll('.spd').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.spd').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.speed = parseFloat(btn.dataset.speed || 1);
        if (this.playing) this._restartTimer();
      });
    });
  }

  play() {
    if (!this.steps.length) return;
    if (this.idx >= this.steps.length - 1) this.idx = 0;
    this.playing = true;
    this.btnPP.textContent = '⏸';
    this._restartTimer();
  }

  pause() {
    this.playing = false;
    this.btnPP.textContent = '▶';
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  _restartTimer() {
    if (this.timer) clearInterval(this.timer);
    const ms = Math.max(350 / this.speed, 25);
    this.timer = setInterval(() => {
      if (this.idx < this.steps.length - 1) {
        this.idx++;
        this.scrubber.value = this.idx;
        this._emit();
      } else {
        this.pause();
      }
    }, ms);
  }

  _step(d) {
    const n = this.idx + d;
    if (n >= 0 && n < this.steps.length) {
      this.idx = n;
      this.scrubber.value = this.idx;
      this._emit();
    }
  }

  _emit() {
    if (!this.steps.length) return;
    const s = this.steps[this.idx];
    if (s && s.t) {
      const d = new Date(s.t * 1000);
      this.timeUTC.textContent = d.toISOString().replace('T', ' ').substring(0, 16) + ' UTC';
    }
    this.timeStep.textContent = `Step ${this.idx + 1} / ${this.steps.length}`;
    if (this.onStepChange) this.onStepChange(this.idx, s);
  }
}
