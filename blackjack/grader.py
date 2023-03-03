#!/usr/bin/env python3

import random, util, collections
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

############################################################
# Problem 1

grader.add_manual_part('1a', 3, description="Written question: value iteration in basic MDP")
grader.add_manual_part('1b', 1, description="Written question: optimal policy in basic MDP")

############################################################
# Problem 2

grader.add_manual_part('2a', 4, description="Written question: define new MDP solver for discounts < 1")

############################################################
# Problem 3

def test3a0():
    mdp1 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    mdp1.computeStates()
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    mdp3 = submission.BlackjackMDP(cardValues=[1, 2, 3], multiplicity=3,
                                   threshold=8, peekCost=1)
    mdp3_startState = mdp3.startState()
    mdp3_preBustState = (6, None, (1, 1, 1))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take'),
        ([((1, None, (2, 3, 3)), 1/3, 0), ((2, None, (3, 2, 3)), 1/3, 0), ((3, None, (3, 3, 2)), 1/3, 0)], mdp3, mdp3_startState, 'Take'),
        ([((7, None, (0, 1, 1)), 1/3, 0), ((8, None, (1, 0, 1)), 1/3, 0), ((9, None, None), 1/3, 0)], mdp3, mdp3_preBustState, 'Take'),
        ([((6, None, None), 1, 6)], mdp3, mdp3_preBustState, 'Quit')
    ]
    for gold, mdp, state, action in tests:
        if not grader.require_is_equal(gold,
                                       mdp.succAndProbReward(state, action)):
            print(('   state: {}, action: {}'.format(state, action)))
grader.add_basic_part('3a-0-basic', test3a0, 5, description="Basic test for succAndProbReward() that covers several edge cases.")

def test3a1():
    mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 7, 9], multiplicity=3,
                                  threshold=30, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)

    grader.require_is_less_than(.001, abs(26.00303030303 - alg.V[startState]))
    grader.require_is_less_than(.001, abs(26.12676767676 - alg.V[(11, 3, (2, 2, 3, 2, 3))]))
    grader.require_is_less_than(.001, abs(24.37499999999 - alg.V[(21, 1, (1, 3, 1, 3, 2))]))

grader.add_basic_part('3a-1-basic', test3a1, 5, description="Test for running ValueIteration on BlackjackMDP.")

def test3a2Hidden():
    mdp = submission.BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                  threshold=40, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)
grader.add_hidden_part('3a-2-hidden', test3a2Hidden, 5, description="Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct.")

############################################################
# Problem 4

def test4a():
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.identityFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback(0, 1, 0, 1)
    grader.require_is_equal(0, rl.getQ(0, -1))
    grader.require_is_equal(0, rl.getQ(0, 1))

    rl.incorporateFeedback(1, 1, 1, 2)
    grader.require_is_equal(0, rl.getQ(0, -1))
    grader.require_is_equal(0, rl.getQ(0, 1))
    grader.require_is_equal(0, rl.getQ(1, -1))
    grader.require_is_equal(1, rl.getQ(1, 1))

    rl.incorporateFeedback(2, -1, 1, 1)
    grader.require_is_equal(1.9, rl.getQ(2, -1))
    grader.require_is_equal(0, rl.getQ(2, 1))

grader.add_basic_part('4a-basic', test4a, 5, max_seconds=10, description="Basic test for incorporateFeedback() using NumberLineMDP.")


def test4aHidden():
    smallMDP = submission.BlackjackMDP(cardValues=[1,5], multiplicity=2, threshold=10, peekCost=1)
    mdp = smallMDP
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                   submission.identityFeatureExtractor,
                                   0.2)
    util.simulate(mdp, rl, 30000)

grader.add_hidden_part('4a-hidden', test4aHidden, 3, max_seconds=3, description="Hidden test for incorporateFeedback(). Run QLearningAlgorithm on smallMDP, then ensure that getQ returns reasonable value.")

grader.add_manual_part('4b', 2, description="Written question: policy comparison for Q-learning vs. value iteration")

# NOTE: this is not a true "test" for grading purposes -- it's worth zero points.  This function exists to run the code
# you need to answer question 4b; this question requires a written response on the assignment, but you will need
# to use the outputs of this test case for your answer.  Check out the implementation of the
# 'simulate_QL_over_MDP' function in submission.py to understand where these stats are coming from.
def run4bHelper():
    print("---- QL versus ValueIteration on small MDP ----")
    submission.simulate_QL_over_MDP(submission.smallMDP, submission.identityFeatureExtractor)
    print("---- QL versus ValueIteration on large MDP ----")
    submission.simulate_QL_over_MDP(submission.largeMDP, submission.identityFeatureExtractor)
grader.add_basic_part('4b-helper', run4bHelper, 0, max_seconds=60, description="Helper function to run Q-learning simulations for question 4b.")

def test4c():
    mdp = submission.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    mdp.computeStates()
    rl = submission.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       submission.blackjackFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))
    grader.require_is_equal(28, rl.getQ((7, None, (0, 1)), 'Quit'))
    grader.require_is_equal(7, rl.getQ((7, None, (1, 0)), 'Quit'))
    grader.require_is_equal(14, rl.getQ((2, None, (0, 2)), 'Quit'))
    grader.require_is_equal(0, rl.getQ((2, None, (0, 2)), 'Take'))
grader.add_basic_part('4c-basic', test4c, 5, max_seconds=10, description="Basic test for blackjackFeatureExtractor.  Runs QLearningAlgorithm using blackjackFeatureExtractor, then checks to see that Q-values are correct.")

grader.add_manual_part('4d', 2, description="Written question: reward comparison for applying policy to baseline and modified MDP")

# NOTE: as in 4b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 4d.
def run4dHelper():
    submission.compare_changed_MDP(submission.originalMDP, submission.newThresholdMDP, submission.blackjackFeatureExtractor)
grader.add_basic_part('4d-helper', run4dHelper, 0, max_seconds=60, description="Helper function to compare rewards when simulating RL over two different MDPs in question 4d.")


############################################################
# Problem 5



grader.add_manual_part('5a', 2, description="Written question: policy comparison for short time horizon MDP versus long time horizon MDP")
# NOTE: as in 4b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 5b.
def run5aHelper():
    submission.compare_MDP_Strategies(submission.short_time, submission.long_time)
grader.add_basic_part('5a-helper', run5aHelper, 0, max_seconds=10, description="Helper function to compare optimal policies over various time horizons.")#

grader.add_manual_part('5b', 2, description="Written question: Ethical frameworks for making time horizon decisions")

grader.add_manual_part('5c', 2, description="Written question: policy comparison for discounted versus non-discounted MDP")
# NOTE: as in 4b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 5d.
def run5cHelper():
    submission.compare_MDP_Strategies(submission.discounted, submission.no_discount)
grader.add_basic_part('5c-helper', run5cHelper, 0, max_seconds=10, description="Helper function to compare optimal policies over various discounts.")#

grader.add_manual_part('5d', 2, description="Written question: Exploring how optimal policies transfer across MDPs")
# NOTE: as in 4b above, this is not a real test -- just a helper function to run some code
# to produce stats that will allow you to answer written question 5d.
def run5dHelper():
    submission.compare_changed_SeaLevelMDP(submission.low_cost, submission.high_cost)
grader.add_basic_part('5d-helper', run5dHelper, 0, max_seconds=10, description="Helper function for exploring how optimal policies transfer across MDPs.")#



grader.grade()
