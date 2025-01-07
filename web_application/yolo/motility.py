import numpy as np

# Thresholds for motility classification
DISPLACEMENT_THRESHOLD = 5  # Minimum displacement in pixels for motility
STRAIGHTNESS_THRESHOLD = 0.8  # Straightness ratio for progressive motility


def calculate_motility_metrics(trajectories):
    motility_data = {
        "total": 0,
        "motile": 0,
        "progressive": 0,
        "immotile": 0
    }

    for track_id, trajectory in trajectories.items():
        motility_data["total"] += 1  # Increment total sperm count


        if len(trajectory) < 2:
            motility_data["immotile"] += 1
            continue

        # Calculate total displacement and path length
        displacements = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))

        # Total displacement is the Euclidean distance between the start and end points 
        total_displacement = np.linalg.norm(np.array(trajectory[-1]) - np.array(trajectory[0]))
        
        path_length = np.sum(displacements)
        straightness = total_displacement / path_length if path_length > 0 else 0

        if total_displacement > DISPLACEMENT_THRESHOLD:
            motility_data["motile"] += 1
            if straightness > STRAIGHTNESS_THRESHOLD:
                motility_data["progressive"] += 1
        else:
            motility_data["immotile"] += 1

    return motility_data