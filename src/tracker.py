import math
import cv2


class CentroidTracker:

    def __init__(
        self,
        max_disappeared=15,
        max_distance=60
    ):

        self.next_object_id = 0

        self.objects = {}

        self.disappeared = {}

        self.max_disappeared = max_disappeared

        self.max_distance = max_distance


    # ========================================================
    # REGISTER NEW OBJECT
    # ========================================================

    def register(self, centroid):

        self.objects[self.next_object_id] = centroid

        self.disappeared[self.next_object_id] = 0

        self.next_object_id += 1


    # ========================================================
    # REMOVE OBJECT
    # ========================================================

    def deregister(self, object_id):

        if object_id in self.objects:

            del self.objects[object_id]

        if object_id in self.disappeared:

            del self.disappeared[object_id]


    # ========================================================
    # UPDATE TRACKING
    # ========================================================

    def update(self, rects):

        # ----------------------------------------------------
        # No detected objects
        # ----------------------------------------------------

        if len(rects) == 0:

            for object_id in list(self.disappeared.keys()):

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):

                    self.deregister(object_id)

            return self.objects


        # ----------------------------------------------------
        # Calculate new centroids
        # ----------------------------------------------------

        input_centroids = []

        for (x, y, w, h) in rects:

            center_x = int(x + w / 2)

            center_y = int(y + h / 2)

            input_centroids.append(
                (center_x, center_y)
            )


        # ----------------------------------------------------
        # No existing objects
        # ----------------------------------------------------

        if len(self.objects) == 0:

            for centroid in input_centroids:

                self.register(centroid)

            return self.objects


        # ----------------------------------------------------
        # Existing objects
        # ----------------------------------------------------

        object_ids = list(
            self.objects.keys()
        )

        object_centroids = list(
            self.objects.values()
        )


        # ----------------------------------------------------
        # Calculate distance matrix
        # ----------------------------------------------------

        distance_matrix = []

        for old_centroid in object_centroids:

            distances = []

            for new_centroid in input_centroids:

                distance = math.sqrt(

                    (old_centroid[0] - new_centroid[0]) ** 2

                    +

                    (old_centroid[1] - new_centroid[1]) ** 2

                )

                distances.append(distance)

            distance_matrix.append(distances)


        # ----------------------------------------------------
        # Match closest objects
        # ----------------------------------------------------

        rows = sorted(

            range(len(distance_matrix)),

            key=lambda r:
            min(distance_matrix[r])

        )


        used_rows = set()

        used_columns = set()


        for row in rows:

            if row in used_rows:

                continue


            column = min(

                range(
                    len(distance_matrix[row])
                ),

                key=lambda c:
                distance_matrix[row][c]

            )


            if column in used_columns:

                continue


            distance = (
                distance_matrix[row][column]
            )


            # Ignore unrealistic movement
            if distance > self.max_distance:

                continue


            object_id = object_ids[row]


            self.objects[object_id] = (
                input_centroids[column]
            )


            self.disappeared[object_id] = 0


            used_rows.add(row)

            used_columns.add(column)


        # ----------------------------------------------------
        # Handle disappeared objects
        # ----------------------------------------------------

        unused_rows = set(
            range(len(object_centroids))
        ).difference(used_rows)


        for row in unused_rows:

            object_id = object_ids[row]

            self.disappeared[object_id] += 1


            if (
                self.disappeared[object_id]
                > self.max_disappeared
            ):

                self.deregister(object_id)


        # ----------------------------------------------------
        # Register new objects
        # ----------------------------------------------------

        unused_columns = set(
            range(len(input_centroids))
        ).difference(used_columns)


        for column in unused_columns:

            self.register(
                input_centroids[column]
            )


        return self.objects


# ============================================================
# DRAW TRACKING INFORMATION
# ============================================================

def draw_tracking(frame, objects):

    for object_id, centroid in objects.items():

        center_x, center_y = centroid


        # Center point
        cv2.circle(

            frame,

            (center_x, center_y),

            5,

            (0, 0, 255),

            -1

        )


        # Object ID
        cv2.putText(

            frame,

            f"ID {object_id}",

            (
                center_x - 25,
                center_y - 10
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 255, 255),

            2,

            cv2.LINE_AA

        )


    return frame