02/11/2026

I analyzed data from one session (M1, 2026-06-18_16-25-09_000) to understand how fusing the thermistor, airflow, and LFP would work. The main goal was to set up a pipeline I can run on the rest of the sessions. 

## Big picture

**1. Each sensor has its own strengths and weaknesses, so combining them should help.**
- Thermistor: cleanest rhythm, but slow.
- Airflow: noisiest signal, locked in timing with the thermistor (highest PLV), but its amplitude varies a lot relative to the thermistor's steadier signal, so their coherence is low. PLV measures timing only; coherence measures timing and amplitude.
- Airflow's ~7 Hz peak is probably not sniffing, since the LFP doesn't share it. This could become a way to tell sniffing apart from movement/noise in future sessions.
- LFP: carries slighly more extra fast activity than the thermistor. Of the 10 channels, 0-3 track breathing best, and 7-9 show the inverse signal

**2. The three sensors share the breathing rhythm most of the time.**

- In clean stretches, all three sensors cycle together with a stable phase difference.
- All three pairs agree well above chance. Thermistor-airflow agrees most, then thermistor-LFP, then airflow-LFP.
- Thermistor and LFP share a strong breathing rhythm in the coherence plot, up to around 7 Hz.

**3. Disagreement is episodic, and not every disagreement is a sensor failure.**
- Some drops coincide with muscle activity (EMG bursts).
- Others don't: around 88 s and 92.5-93 s, all three signals lose strength together with no EMG activity. Each dip follows an unusually large airflow signal, which could be a sigh followed by a pause, or drinking followed by a swallow. Video recording shows wheel movement after the signal drop, does this mean it could be taking a long sniff? 

**4. Phase as a common format.**
- LFP and airflow add occasional extra short cycles from fast activity.
- When a signal's amplitude drops, its phase becomes unreliable. In messy stretches, the phase difference jumps erratically.
- The shared breathing rhythm stops around 7 Hz, maybe data over more sessions can reveal more in higher frequencies. 



**Questions**
1. Is there more info we can uncover from the LFP?   
2. No clear pattern during messy phases is showing given one session, given the massive amounts of data, what are next steps that I could take to generalize this pipeline?
3. 

**Next steps**
1. Run this pipeline on a few more sessions to build a baseline for all the sensors.
2. Identify all points of signal failure and determine reason.
3. 





Coherence plot 
    - Which frequencies do they share?
    - shifting one sensor by 60s so "happening at the same time" isn't a factor. 
    1. At every frequency, cut both signals into chunks (around 4 secs). Then turn the signal into two vectors, keeping in mind vector length (amplitude) and direction. 
    2. new length = lengthA * lengthB;   direction = dirA - dirB (phase difference)
    3. average all the vectors then scale this formula to between 0-1
        coherence = (length of average combined vectors)^2/((average of lengthA)^2(average of lengthB)^2)
        1: every chunk has the same phase difference; 0: no consistent relationship

    - Thermistor and LFP share a breathing rhythm. 
    - Shared breathing rhythm stops at around 7Hz, will need more sessions to eestablish a pattern between any two signals 
    - Airflow's peak probably isn't sniffing since the LFP should spike as well (could be way to distinguish sniffing from movement/noise in future sessions)
    - Note: Thermistor and airflow coherence is low despite highest PLV of all three pairs 


Phase plots
    1. Bandpass 1-15Hz and Hilbert transform, so we can view all three sensors under the same units.
       - 0 is signal peak and +-pi is trough
    
    - What I learned:
        - Clean cycle: All three sensors cycle together during clean phases with stable phase difference. With LFP and airflow showing occasional short cycles. 
        - Messy cycle: phase difference spikes irrationally, low amplitude makes phase unreliable. 

PSL
    - When do the signals agree?
    - phase lock values in 5-s windows, compared against a chance level using EMG as                      reference for disagreements 

    1. Take the phase differnce at every value and normalize to vector of length 1
    2. Add them tip to tail then average
    3. This gives us a plv value for each second so we do this for every second which forms the curve 
    
    What I learned:
      - Thermistor-airflow agrees most, then thermistor-LFP, then LFP-airflow 
      - Disagreement is episodic, some coincide with EMG bursts. 
      - Note a dip around 88 with no EMG spike

Breathing Pause plot
    - Zoom in on 88s - 100 and compare with typical signal strength
    - Hilbert transform gives amplitude as well as cycle position
    - What I learned:
        - Dips from all three signals with no coinciding EMG spike and a spike in airfflow before each could suggest licking/drinking before swallowing(the dip) or a sigh? 
        - Some singal losses are intentional and not necessary singal failures
        - Note: our filter ranges from 1-15Hz so a long breath for over a second could be untracked  

