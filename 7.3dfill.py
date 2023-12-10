class SeedFill3D:
    def __init__(self, input_array: np.ndarray, start_area_id: int = 0):
        self.input_array = input_array
        self.state_array = np.full_like(input_array, -2)
        self.x, self.y, self.z = self.input_array.shape
        self.waiting_queue = None
        self.area_id = start_area_id
        self.around = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])

    def _check_around(self, location, classify):
        self.state_array[location] = self.area_id
        location = np.array(location)
        for displacement in self.around:
            temp_location = tuple(location + displacement)
            try:
                if self.state_array[temp_location] == -2 and self.input_array[temp_location] == classify:
                    self.waiting_queue.append(temp_location)
                    self.state_array[temp_location] = -1
            except IndexError:
                None

    def result(self) -> np.ndarray:
        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    loca = (i, j, k)
                    if self.state_array[loca] == -2:
                        self.waiting_queue = [loca]
                        classify = self.input_array[loca]
                        while self.waiting_queue:
                            location = self.waiting_queue.pop()
                            if self.input_array[location] == classify:
                                self._check_around(location, classify)
                        self.area_id += 1
        return self.state_array