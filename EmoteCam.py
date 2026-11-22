from collections import deque
import time
import tkinter as tk
import cv2
from deepface import DeepFace
from PIL import Image, ImageTk


NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}

QUESTIONS = [
    {
        "prompt": "1. What is 12 + 8?",
        "answer": "20",
        "hint": "Add the tens place first (10 + 8 = 18), then add the remaining 2."
    },
    {
        "prompt": "2. In Python, what keyword defines a function?",
        "answer": "def",
        "hint": "It is a three-letter keyword followed by the function name and parentheses."
    },
    {
        "prompt": "3. What is 15 * 4?",
        "answer": "60",
        "hint": "Multiply 10 * 4 = 40 and 5 * 4 = 20, then sum the two results."
    }
]


class SensingLayer:
    def __init__(
        self,
        camera_index = 0,
        sample_interval = 1.5,
        detector_backend = "opencv",
        sad_threshold = 65.0
    ):
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.detector_backend = detector_backend
        self.sad_threshold = sad_threshold
        self.cap = None
        self.last_sample_time = 0.0

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to access the webcam.")

    def close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def read_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def should_sample(self, current_time):
        return (current_time - self.last_sample_time) >= self.sample_interval

    def classify_expression(self, frame, current_time):
        self.last_sample_time = current_time
        try:
            results = DeepFace.analyze(
                img_path = frame,
                actions = ["emotion"],
                detector_backend = self.detector_backend,
                enforce_detection = False,
                silent = True
            )
            if isinstance(results, list) and len(results) > 0:
                first = results[0]
                dominant = first.get("dominant_emotion")
                emotions = first.get("emotion", {})
                region = first.get("region", {})
                sad_score = float(emotions.get("sad", 0.0))

                if sad_score >= self.sad_threshold:
                    selected = "sad"
                elif isinstance(dominant, str):
                    selected = dominant.lower()
                else:
                    selected = None

                top_score = (
                    sad_score
                    if selected == "sad"
                    else float(emotions.get(selected, 0.0)) if selected else 0.0
                )
                print(
                    f"[Sample] selected={selected} ({top_score:.1f}%) "
                    f"dominant={dominant} sad={sad_score:.1f}% region={region}"
                )
                return selected, region
        except Exception as exc:
            print(f"[Error] DeepFace.analyze failed: {exc}")
        return None, None


class StateInferenceLayer:
    def __init__(
        self,
        window_size = 5,
        frustration_ratio_threshold = 0.8,
        min_frustration_samples = 5
    ):
        self.window = deque(maxlen = window_size)
        self.frustration_ratio_threshold = frustration_ratio_threshold
        self.min_frustration_samples = min_frustration_samples
        self.in_task_log = []

    def reset_window(self):
        self.window.clear()

    def update(self, emotion):
        if emotion is not None:
            self.window.append(emotion)

    def infer_state(self):
        if not self.window:
            state = "focused"
            ratio = 0.0
        else:
            negative_count = sum(
                1 for item in self.window if item in NEGATIVE_EMOTIONS
            )
            ratio = negative_count / len(self.window)

            if (
                len(self.window) >= self.min_frustration_samples
                and ratio >= self.frustration_ratio_threshold
            ):
                state = "frustrated"
            elif "happy" in self.window:
                state = "engaged"
            else:
                state = "focused"

        self.in_task_log.append((state, ratio))
        return state, ratio


class AdaptationLayer:
    def decide_adaptation(
        self,
        inferred_state,
        negative_ratio,
        wrong_attempts
    ):
        activate_hint = (inferred_state == "frustrated") or (wrong_attempts >= 2)
        if inferred_state == "frustrated":
            status_line = (
                f"Adaptation: Frustration detected ({negative_ratio:.0%} "
                "recent negative frames) -> Hint activated."
            )
        elif inferred_state == "engaged":
            status_line = (
                "Adaptation: Engaged -> Keep steady pace and check your answer."
            )
        else:
            status_line = (
                "Adaptation: Focused -> Work through the question step by step."
            )

        return status_line, activate_hint


class AdaptiveTestScreen:
    def __init__(
        self,
        questions,
        adaptation_layer,
        on_question_advance = None,
        title = "Adaptive Learning Test - PoC"
    ):
        self.questions = questions
        self.adaptation_layer = adaptation_layer
        self.on_question_advance = on_question_advance
        self.current_index = 0
        self.wrong_attempts = 0
        self.correct_count = 0
        self.hint_unlocked_question_idx = None
        self.completed = False
        self.closed = False
        self.feedback_text = ""
        self.photo = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x600")
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.video_label = tk.Label(self.root)
        self.video_label.pack(
            padx = 12,
            pady = (10, 6)
        )

        self.status_label = tk.Label(
            self.root,
            text = "",
            font = ("Arial", 10, "italic"),
            fg = "#1f4e78",
            wraplength = 560,
            justify = tk.LEFT
        )
        self.status_label.pack(
            padx = 16,
            pady = 4
        )

        self.question_label = tk.Label(
            self.root,
            text = "",
            font = ("Arial", 13, "bold"),
            wraplength = 560,
            justify = tk.LEFT
        )
        self.question_label.pack(
            padx = 16,
            pady = (8, 6)
        )

        self.answer_var = tk.StringVar()
        self.answer_entry = tk.Entry(
            self.root,
            textvariable = self.answer_var,
            font = ("Arial", 12),
            width = 30
        )
        self.answer_entry.pack(pady = 4)
        self.answer_entry.bind(
            "<Return>",
            self._on_submit_event
        )

        self.submit_btn = tk.Button(
            self.root,
            text = "Submit Answer",
            font = ("Arial", 11),
            command = self.submit_answer
        )
        self.submit_btn.pack(pady = 6)

        self.feedback_label = tk.Label(
            self.root,
            text = "",
            font = ("Arial", 11),
            wraplength = 560,
            justify = tk.LEFT
        )
        self.feedback_label.pack(
            padx = 16,
            pady = 8
        )

    def _on_submit_event(self, _event):
        self.submit_answer()

    def submit_answer(self):
        if self.completed:
            return
        user_ans = self.answer_var.get().strip().lower()
        expected = self.questions[self.current_index]["answer"].strip().lower()

        if user_ans == expected:
            self.correct_count += 1
            self.current_index += 1
            self.wrong_attempts = 0
            self.hint_unlocked_question_idx = None
            self.answer_var.set("")
            if callable(self.on_question_advance):
                self.on_question_advance()
            if self.current_index >= len(self.questions):
                self.completed = True
                self.feedback_text = "Test completed."
                self.answer_entry.config(state = tk.DISABLED)
                self.submit_btn.config(state = tk.DISABLED)
            else:
                self.feedback_text = "Correct. Moving to the next question."
        else:
            self.wrong_attempts += 1
            self.feedback_text = (
                f"Not quite (attempts on this question: {self.wrong_attempts})."
            )

    def update_view(
        self,
        frame,
        inferred_state,
        negative_ratio,
        in_task_summary
    ):
        if self.closed:
            return
        try:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            resized = cv2.resize(
                rgb_frame,
                (480, 240)
            )
            image = Image.fromarray(resized)
            self.photo = ImageTk.PhotoImage(image = image)
            self.video_label.config(image = self.photo)

            if self.completed:
                self.status_label.config(text = "Session Complete")
                self.question_label.config(
                    text = (
                        f"Score: {self.correct_count} / {len(self.questions)}"
                    )
                )
                self.feedback_label.config(
                    text = (
                        "Post-task summary: Compare your final score with "
                        f"in-task states ({in_task_summary}).\n"
                        "Close the window to exit."
                    )
                )
            else:
                question = self.questions[self.current_index]
                status_line, activate_hint = (
                    self.adaptation_layer.decide_adaptation(
                        inferred_state,
                        negative_ratio,
                        self.wrong_attempts
                    )
                )
                if activate_hint:
                    self.hint_unlocked_question_idx = self.current_index

                show_hint = (
                    self.hint_unlocked_question_idx == self.current_index
                )
                hint_block = (
                    f"\nHint: {question['hint']}" if show_hint else ""
                )
                self.status_label.config(text = status_line)
                self.question_label.config(text = question["prompt"])
                self.feedback_label.config(
                    text = f"{self.feedback_text}{hint_block}".strip()
                )

            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def format_in_task_summary(log):
    if not log:
        return "no samples recorded"
    counts = {}
    for state, _ in log:
        counts[state] = counts.get(state, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def run_poc(
    camera_index = 0,
    sample_interval = 1.5,
    window_size = 5
):
    sensing = SensingLayer(
        camera_index = camera_index,
        sample_interval = sample_interval,
        detector_backend = "opencv",
        sad_threshold = 65.0
    )
    inference = StateInferenceLayer(
        window_size = window_size,
        frustration_ratio_threshold = 0.8,
        min_frustration_samples = 5
    )
    adaptation = AdaptationLayer()

    latest_emotion = "none"
    current_state = "focused"
    negative_ratio = 0.0
    last_region = None

    def handle_question_advance():
        nonlocal current_state, negative_ratio
        inference.reset_window()
        current_state = "focused"
        negative_ratio = 0.0

    screen = AdaptiveTestScreen(
        QUESTIONS,
        adaptation,
        on_question_advance = handle_question_advance
    )

    sensing.open()
    try:
        while not screen.closed:
            now = time.time()
            frame = sensing.read_frame()
            if frame is None:
                break

            if sensing.should_sample(now):
                detected, region = sensing.classify_expression(frame, now)
                if region:
                    last_region = region
                if detected is not None:
                    latest_emotion = detected
                    inference.update(detected)
                    current_state, negative_ratio = inference.infer_state()

            if last_region:
                x = last_region.get("x", 0)
                y = last_region.get("y", 0)
                w = last_region.get("w", 0)
                h = last_region.get("h", 0)
                if w > 0 and h > 0:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 200, 0),
                        2
                    )

            overlay_text = f"Emotion: {latest_emotion} | State: {current_state}"
            cv2.putText(
                frame,
                overlay_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            summary = format_in_task_summary(inference.in_task_log)
            screen.update_view(
                frame,
                current_state,
                negative_ratio,
                summary
            )
    finally:
        sensing.close()
        screen.close()


if __name__ == "__main__":
    run_poc(
        camera_index = 0,
        sample_interval = 1.5,
        window_size = 5
    )