# ONNX Runtime
import onnxruntime as ort
import numpy as np

###############################################################################
# Class: IsaacLab
#   Loads an ONNX model and provides a method to run inference.
###############################################################################
class IsaacLab:
    def __init__(self, model_path: str):
        """
        Initialize the ONNX runtime session with the given model_path.
        """
        self.ort_session = ort.InferenceSession(model_path)
        print(f"[IsaacLab] Loaded ONNX model from: {model_path}")

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Run inference given an observation, return the model's output action.
        observation should be shape (1, input_dim).
        """
        #   6 joint_pos + 6 joint_vel + 7 pose + 6 last_actions = 25
        if observation.shape[1] != 25:
            raise ValueError(f"[IsaacLab] Expected observation shape (1, 25), got {observation.shape}")

        outputs = self.ort_session.run(None, {"obs": observation})

        # Suppose the model returns one array for the 6D action
        action = outputs[0].squeeze()
        return action