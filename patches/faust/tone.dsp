declare name "AmbitionTone";

// Simple high-quality Faust oscillator layer meant to be shaped by YAML
// automation and Python-side envelopes/effects.
freq = hslider("freq", 440, 20, 16000, 1);
gain = hslider("gain[unit:dB]", -12, -120, 12, 0) : ba.db2linear;
process = os.osc(freq) * gain <: _, _;
