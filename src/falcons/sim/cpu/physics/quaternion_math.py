import numpy as np

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]"""
    q = np.zeros(4)
    q[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return q

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion"""
    return q / np.linalg.norm(q)

def quat_rotate_vector(q: np.ndarray, v: np.ndarray, i_to_b: bool = True) -> np.ndarray:
    """Rotate a vector using quaternion rotation"""
    q_conj = quat_conjugate(q)
    v_quat = np.concatenate(([0], v))
    
    if i_to_b:
        result = quat_multiply(quat_multiply(q_conj, v_quat), q)
    else:
        result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    
    return result[1:]