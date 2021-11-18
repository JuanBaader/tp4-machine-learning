import math


class Perceptron:

    def __init__(self, i, j, n, example):
        self.i = i
        self.j = j
        self.n = n
        self.w = []
        self.hits = 0
        cols = example.columns
        for i in range(n):
            self.w.append(example[cols[i]])

    def get_distance(self, data):
        distance = 0
        for i in range(self.n):
            distance += math.pow(self.w[i] - data[i], 2)

        return math.sqrt(distance)

    def train(self, data, learning_rate, v):
        for i in range(self.n):
            self.w[i] = self.w[i] + v * learning_rate * data[i]

    def hit(self):
        self.hits += 1
