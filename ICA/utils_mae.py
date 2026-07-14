import re
import mne

def _has_trigger_code(description, trigger_code):
    numbers = re.findall(r"\d+", description)
    return str(trigger_code) in numbers


def _extract_blocks(raw_data, start_code=4, end_code=8, require_S=True, pad_end=50.0):
    """
    Extract list of Raw segments from start_code (S4) to end_code (S8) + pad_end seconds.
    Only annotations containing "S" are considered when require_S is True.
    """
    starts = [o for o, d in zip(raw_data.annotations.onset, raw_data.annotations.description)
              if _has_trigger_code(d, start_code) and (("S" in d) if require_S else True)]
    ends = [o for o, d in zip(raw_data.annotations.onset, raw_data.annotations.description)
            if _has_trigger_code(d, end_code) and (("S" in d) if require_S else True)]

    if not starts or not ends:
        return []

    segments = []
    for s, e in zip(starts, ends):
        t0 = float(s)
        t1 = min(float(e) + float(pad_end), raw_data.times[-1])
        segments.append(raw_data.copy().crop(tmin=t0, tmax=t1))
    return segments

def _concat_first_n(segments, n):
    """
    Concatenate first n Raw segments. If fewer than n available, concatenate all.
    Returns None if segments empty.
    """
    if not segments:
        return None
    n_use = min(n, len(segments))
    return mne.concatenate_raws(segments[:n_use])


def crop_eeg_data(raw_data, task_name, subject_id=None, session_id=None):
    """
    Cropping function for specific EEG tasks with Sanity Checks.
    """
    
    # Safety check: Ensure raw_data has annotations
    if len(raw_data.annotations) == 0:
        print(f"{task_name}: No annotations found. Returning original data.")
        return raw_data

    # --- TASK RSpre/RSpost ---
    if task_name in ("RSpre", "RSpost"):
        trigger_onset = None
        for onset, desc in zip(raw_data.annotations.onset, raw_data.annotations.description):
            if _has_trigger_code(desc, 15):
                trigger_onset = float(onset)
                break
        
        if trigger_onset is not None:
            crop_start = trigger_onset
            crop_end = min(crop_start + 300.0, raw_data.times[-1])
            raw_data.crop(tmin=crop_start, tmax=crop_end)
            
            actual_duration = raw_data.times[-1] - raw_data.times[0]
            print(f"{task_name}: Cropped. Duration: {actual_duration:.2f}s (Target: 300s).")
        else:
            print(f"{task_name}: Trigger 15 not found. No crop applied.")
        return raw_data 

    # --- TASK B ---
    elif task_name == 'B':
        t_start = None
        t_end = None
        
        for onset, desc in zip(raw_data.annotations.onset, raw_data.annotations.description):
            if _has_trigger_code(desc, 4) and t_start is None:
                t_start = float(onset)
            if _has_trigger_code(desc, 8):
                t_end = float(onset) + 50.0
        
        if t_start is not None and t_end is not None:
            t_end = min(t_end, raw_data.times[-1])
            raw_data.crop(tmin=t_start, tmax=t_end)
            
            actual_duration = raw_data.times[-1] - raw_data.times[0]
            print(f"{task_name}: Cropped from Trig 4 to 8+50s. Duration: {actual_duration:.2f}s.")
        else:
            print(f"{task_name}: Triggers 4 or 8 missing. No crop applied.")
        return raw_data

    # --- TASK RSstim ---
    elif task_name == 'RSstim':
        segments = []
        for onset, desc in zip(raw_data.annotations.onset, raw_data.annotations.description):
            if _has_trigger_code(desc, 15) and "S" in desc:
                t0 = float(onset)
                t1 = min(t0 + 270.0, raw_data.times[-1])
                segments.append(raw_data.copy().crop(tmin=t0, tmax=t1))
        
        if segments:
            raw_concatenated = mne.concatenate_raws(segments)
            
            # Use raw_concatenated here to check the NEW duration
            duration = raw_concatenated.times[-1] - raw_concatenated.times[0]
            print(f"{task_name}: Found {len(segments)} segments. Total Duration: {duration:.2f}s.")
            return raw_concatenated
        
        else:
            print(f"{task_name}: No Trigger 15 found. No concatenation applied.")
        return raw_data
    
   # --- TASK-TASK / TASK-TASK_v2 ---
    elif task_name in ("task", "task_v2"):
        # Find all S4 (starts) and S8 (ends)
        starts = [o for o, d in zip(raw_data.annotations.onset, raw_data.annotations.description) if _has_trigger_code(d, 4) and "S" in d]
        ends = [o for o, d in zip(raw_data.annotations.onset, raw_data.annotations.description) if _has_trigger_code(d, 8) and "S" in d]

        # Check if we have matching pairs
        if len(starts) == 0 or len(ends) == 0:
            print(f"{task_name}: Triggers missing.")
            return raw_data

        # Create all possible segments found in the file
        all_found = []
        for s, e in zip(starts, ends):
            t0 = float(s)
            t1 = min(float(e) + 50.0, raw_data.times[-1])
            all_found.append(raw_data.copy().crop(tmin=t0, tmax=t1))

        # Special-case behaviour for subject 15, session 1:
        if str(subject_id) == "15" and str(session_id) == "1":
            # For task-task -> concatenate first 3 blocks (if available)
            if task_name == "task":
                n3 = min(3, len(all_found))
                if n3 == 0:
                    print(f"{task_name}: special-case -> no blocks available for first3.")
                    return raw_data
                first3 = mne.concatenate_raws(all_found[:n3])
                print(f"{task_name}: special-case -> concatenated first {n3} blocks (duration: {first3.times[-1] - first3.times[0]:.2f}s).")
                return first3

            # For task_v2 -> concatenate first 6 blocks (if available)
            elif task_name == "task_v2":
                n6 = min(6, len(all_found))
                if n6 == 0:
                    print(f"{task_name}: special-case -> no blocks available for first6.")
                    return raw_data
                first6 = mne.concatenate_raws(all_found[:n6])
                print(f"{task_name}: special-case -> concatenated first {n6} blocks (duration: {first6.times[-1] - first6.times[0]:.2f}s).")
                return first6
        # Normal behaviour (all other subjects/sessions): concatenate all found blocks
        if all_found:
            raw_concatenated = mne.concatenate_raws(all_found)
            duration = raw_concatenated.times[-1] - raw_concatenated.times[0]
            print(f"{task_name}: Final duration: {duration:.2f}s (concatenated {len(all_found)} blocks).")
            return raw_concatenated

        return raw_data

    # Return original if task_name doesn't match anything
    return raw_data
