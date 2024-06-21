import csv
import math
import operator


def calculateEuclideanDistance(variable1, variable2):
    # Ensure both vectors are of the same length
    if len(variable1) != len(variable2):
        raise ValueError("Vectors must be of the same length")

    distance = 0
    for i in range(len(variable1)):
        distance += pow(variable1[i] - variable2[i], 2)
    return math.sqrt(distance)


def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []
    for instance in training_feature_vector:
        dist = calculateEuclideanDistance(testInstance[:-1], instance[:-1])
        distances.append((instance, dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors


def responseOfNeighbors(neighbors):
    votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1

    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def loadDataset(filename):
    feature_vector = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            feature_vector.append([float(val) for val in row])
    return feature_vector


def main(training_data, test_data):
    training_feature_vector = loadDataset(training_data)
    test_feature_vector = loadDataset(test_data)
    k = 3  # K value for KNN

    predictions = []
    for instance in test_feature_vector:
        neighbors = kNearestNeighbors(training_feature_vector, instance, k)
        result = responseOfNeighbors(neighbors)
        predictions.append(result)

    return predictions


if __name__ == "__main__":
    training_data = 'training.data'
    test_data = 'test.data'
    predictions = main(training_data, test_data)
    print("Predictions:", predictions)
