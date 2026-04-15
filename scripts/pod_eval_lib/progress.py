import json


def write_phase(progress_path, students, phase, teacher_done=None, **extra):
    try:
        data = {
            "phase": phase,
            "students": students,
            "students_total": len(students),
            "prompts_total": extra.get("prompts_total", 0),
            "teacher_prompts_done": teacher_done,
            "completed": extra.get("completed", []),
            "current": extra.get("current", None),
        }
        with open(progress_path, "w") as handle:
            json.dump(data, handle)
    except Exception:
        pass
