# Clothes recognition with neural networks
# Suit recognition

from newConx import *
from defClothes import *

class SuitRecognizer(BackpropNetwork):
    """
    A specialied backprop network for classifying face images for
    the position of head.
    """
    def classify(self, output):
        """
        This ensures that that output layer is the correct size for this
        task, and then tests whether the output value is within
        tolerance of 1 (meaning sunglasses are present) or 0 (meaning
        they are not).
        """
        #assert len(output) != 4, 'invalid output pattern'
        # counter = 0
        # for i in output:
        #     if i > self.tolerance:
        #         counter += 1
        if len(output) != len(article_text):
            return '???'
        for i in range(len(article_text)):
            if output[i] > (1 - self.tolerance):
                return article_text[i]
        # if output[0] > (1 - self.tolerance):
        #     return 'suits'
        # elif output[1] > (1 - self.tolerance):
        #     return 'shorts'
        # elif output[2] > (1 - self.tolerance):
        #     return 'bottoms'
        # elif output[3] > (1 - self.tolerance):
        #     return 'shirts'
        # elif output[4] > (1 - self.tolerance):
        #     return 'tops'
        # elif output[5] > (1 - self.tolerance):
        #     return 'vests' 
        # elif output[6] > (1 - self.tolerance):
        #     return 'tshirts' 
        # else:
        return '???'

    def evaluate(self):
        """
        For the current set of inputs, tests each one in the network to
        determine its classification, compares this classification to
        the targets, and computes the percent correct.
        """
        if len(self.inputs) == 0:
            print 'no patterns to evaluate'
            return
        correct = 0
        wrong = 0
        for i in range(len(self.inputs)):
            pattern = self.inputs[i]
            target = self.targets[i]
            output = self.propagate(input=pattern)

            networkAnswer = self.classify(output)
            correctAnswer = self.classify(target)
            if networkAnswer == correctAnswer:
                correct = correct + 1
            else:
                wrong = wrong + 1
                print 'network classified image #%d (%s) as %s' % \
                      (i, correctAnswer, networkAnswer)
        total = len(self.inputs)
        correctPercent = float(correct) / total * 100
        wrongPercent = float(wrong) / total * 100
        print '%d patterns: %d correct (%.1f%%), %d wrong (%.1f%%)' % \
              (total, correct, correctPercent, wrong, wrongPercent)

        
#create the network
n = SuitRecognizer()


n.addLayers(w * h * 4, len(article_text) *2, len(article_text))

#get the input and target data
rootname = "inputs/"
# inputs:
# "clothes29*23-input.dat"
# "tbs-30-144*108-input.dat"
# "test-inputs.dat"
n.loadInputsFromFile("inputs/train-test/train0426-inputs.dat")

# outputs:
# "top-bottom-suit-targets.dat" suit: 1 0 0 bottom: 0 1 0 top: 0 0 1
# "tbs-30-144*108-targets.dat" suit: 1 0 0 bottom: 0 1 0 top: 0 0 1
# test-targets.dat
n.loadTargetsFromFile("inputs/train-test/train0426-targets.dat")

#set the training parameters
n.setEpsilon(0.3)
n.setMomentum(0.1)
n.setReportRate(1)
n.setTolerance(0.2)

#create the visualization windows
n.showActivations('input', shape=(w,h), scale= scale)
n.showActivations('hidden', scale=100)
n.showActivations('output', scale=100)
n.showWeights('hidden', shape=(w,h), scale= scale)

# use 80% of dataset for training, 20% for testing
n.splitData(100)

print "Clothes recognition network is set up"

# Type the following at the python prompt:
#
# >>> n.showData()
# >>> n.showPerformance()
# >>> n.evaluate()
# >>> n.train()
# >>> n.showPerformance()
# >>> n.evaluate()
# >>> n.swapData()
# >>> n.showPerformance()
# >>> n.evaluate()

