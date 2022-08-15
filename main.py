import numpy as np

class Boid():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.position = np.array([self.x,self.y])
        vec = (np.random.rand(2) - 0.5)*10
        self.velocity = vec

        vec = (np.random.rand(2) - 0.5)/2
        self.acceleration = vec
        self.max_force = 0.3
        self.max_speed = 5
        self.perception = 100

        self.width = width
        self.height = height



    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed
        self.acceleration = np.zeros(2)


    def apply_behaviour(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)
        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

    def edges(self):
        if self.position[0] > self.width:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = self.width

        if self.position[1] > self.height:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = self.height

    def align(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                avg_vector += boid.velocity
                total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity

        return steering

    def cohesion(self, boids):
        steering = np.zeros(2)
        total = 0
        center_of_mass = np.zeros(2)
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                center_of_mass += boid.position
                total += 1
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity
            if np.linalg.norm(steering)> self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

    def separation(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for boid in boids:
            distance = np.linalg.norm(boid.position - self.position)
            if self.position.all() != boid.position.all() and distance < self.perception:
                diff = self.position - boid.position
                diff /= distance
                avg_vector += diff
                total += 1
        if total > 0:
            avg_vector /= total
            if np.linalg.norm(steering) > 0:
                avg_vector = (avg_vector / np.linalg.norm(steering)) * self.max_speed
            steering = avg_vector - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

width = 1000
height = 1000

flock = [Boid(np.random.rand(1)*1000, np.random.rand(1)*1000, width, height) for _ in range(10)]

def draw():
    global flock

    for boid in flock:
        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        print(boid.position)
        #boid.show()

def run(time = 200): 

    for t in range(time): 
        draw()

    print("done")

run()








