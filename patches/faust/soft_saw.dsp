declare name "AmbitionSoftSaw";

freq = hslider("freq", 220, 20, 16000, 1);
gain = hslider("gain[unit:dB]", -14, -120, 12, 0) : ba.db2linear;
// Blend saw with sine to reduce harshness before Pedalboard processing.
body = (os.sawtooth(freq) * 0.55 + os.osc(freq) * 0.45) * gain;
process = body <: _, _;
