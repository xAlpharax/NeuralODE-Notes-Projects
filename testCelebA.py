from utils.CelebADataset import CelebA

celeb = CelebA()

train = celeb.load()
test = celeb.testSet()

total_size = len(train)

print("Train Data: {}".format(train[0].shape))
print("Test Data: {}".format(test[0].shape))