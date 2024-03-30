def calculate_midpoint(point1, point2):
    return (point1 + point2) / 2


def find_triangle_midpoints(tri, points):
    midpoints_dict = {}

    for simplex in tri.simplices:
        # Calculate midpoints for current triangle
        midpoints = [
            calculate_midpoint(points[simplex[i]], points[simplex[(i + 1) % 3]])
            for i in range(3)
        ]
        midpoint_tuples = [tuple(midpoint) for midpoint in midpoints]

        # Store opposite midpoints
        for i in range(3):
            opposite_midpoint1 = midpoint_tuples[(i + 1) % 3]
            opposite_midpoint2 = midpoint_tuples[(i + 2) % 3]
            current_midpoint = midpoint_tuples[i]

            if current_midpoint in midpoints_dict:
                midpoints_dict[current_midpoint].add(
                    (opposite_midpoint1, opposite_midpoint2)
                )
            else:
                midpoints_dict[current_midpoint] = {
                    (opposite_midpoint1, opposite_midpoint2)
                }

    # Convert sets back to lists for easier use
    for midpoint in midpoints_dict:
        midpoints_dict[midpoint] = list(midpoints_dict[midpoint])

    return midpoints_dict


def create_indices_dict(coords_dict):
    # Step 1: Extract all unique coordinates
    unique_coords = set()
    for coord1, coord_pairs in coords_dict.items():
        unique_coords.add(coord1)
        for coord2, coord3 in coord_pairs:
            unique_coords.add(coord2)
            unique_coords.add(coord3)

    # Step 2: Create a mapping of coordinates to indices
    coord_to_index = {coord: i for i, coord in enumerate(unique_coords)}

    # Step 3: Build the new dictionary with indices
    indexed_dict = {}
    for coord1, coord_pairs in coords_dict.items():
        indexed_pairs = [
            (coord_to_index[coord2], coord_to_index[coord3])
            for coord2, coord3 in coord_pairs
        ]
        indexed_dict[coord_to_index[coord1]] = indexed_pairs

    return indexed_dict
