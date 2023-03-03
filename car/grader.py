#!/usr/bin/env python3

import random, sys

from engine.const import Const
import graderUtil
import util
import collections
import copy

graderUtil.TOLERANCE = 1e-3
grader = graderUtil.Grader()
submission = grader.load('submission')

# General Notes:
# - Unless otherwise specified, all parts time out in 1 second.

############################################################
# Problem 1: Emission probabilities

def test1a():
    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.030841805296, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00073380582967, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0269846478431, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0129150762582, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.00000261584106271, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.000924335357194, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0295673460685, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.000102360275238, ei.belief.getProb(5, 9))

grader.add_basic_part('1a-0-basic', test1a, 2, description="1a basic test for emission probabilities")

def test1a_1(): # test whether they put the pdf in the correct order
    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # be super rude to them! You can't swap a and c now!
      return a + b
    util.pdf = pdf

    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.012231949648, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00982248065925, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0120617259453, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0152083233155, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.0159738258744, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00989135100651, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0122435075636, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.018212043367, ei.belief.getProb(5, 9))
    util.pdf = oldpdf # replace the old pdf

grader.add_basic_part('1a-1-basic', test1a_1, 2, description="1a test ordering of pdf")

def test1a_2():
    random.seed(10)

    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1

    N = 50
    p_values = []
    for i in range(N):
      a = int(random.random() * 300)
      b = int(random.random() * 5)
      c = int(random.random() * 300)

      ei.observe(a, b, c)

      for d in range(10):
        for e in range(10):
          p_values.append(ei.belief.getProb(d, e))

grader.add_hidden_part('1a-2-hidden', test1a_2, 3, description="1a advanced test for emission probabilities")

############################################################
# Problem 2: Transition probabilities

def test2a():
    ei = submission.ExactInference(30, 13)
    ei.elapseTime()
    grader.require_is_equal(0.0105778989624, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00250560512469, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0165024135157, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0178755550388, ei.belief.getProb(8, 4))

    ei.elapseTime()
    grader.require_is_equal(0.0138327373012, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00257237608713, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0232612833688, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0176501876956, ei.belief.getProb(8, 4))

grader.add_basic_part('2a-0-basic', test2a, 2, description="test correctness of elapseTime()")

def test2a_1i(): # stress test their elapseTime
    A = 30
    B = 30
    random.seed(15)
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))


grader.add_hidden_part('2a-1i-hidden', test2a_1i, 2, description="advanced test for transition probabilities, strict time limit", max_seconds=5)

def test2a_1ii(): # stress test their elapseTime, making sure they didn't specifically use lombard
    random.seed(15)

    oldworld = Const.WORLD
    Const.WORLD = 'small' # well... they may have made it specific for lombard

    A = 30
    B = 30
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 40
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))
    Const.WORLD = oldworld # set it back to what's likely lombard


grader.add_hidden_part('2a-1ii-hidden', test2a_1ii, 1, description="2a test for transition probabilities on other maps, loose time limit", max_seconds=20)

def test2a_2(): # let's test them together! Very important
    # This assumes the rest of the tests will be run on lombard
    Const.WORLD = 'lombard' # set it to lombard in case the previous test times out
    random.seed(20)

    A = 30
    B = 30
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()

      a = int(random.random() * 5 * A)
      b = int(random.random() * 5)
      c = int(random.random() * 5 * A)

      ei.observe(a, b, c)
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))


grader.add_hidden_part('2a-2-hidden', test2a_2, 2, description="advanced test for emission AND transition probabilities, strict time limit", max_seconds=5)


############################################################
# Problem 3: Particle filtering

def test3a_0():
    random.seed(3)

    pf = submission.ParticleFilter(30, 13)

    pf.observe(555, 193, 800)
    grader.require_is_equal(0.02, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.04, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.94, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(8, 4))

    pf.observe(525, 193, 830)
    grader.require_is_equal(0.0, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(1.0, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(8, 4))


grader.add_basic_part('3a-0-basic', test3a_0, 2, description="3a basic test for PF observe")

def test3a_1():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.require_is_equal(69, len([k for k, v in list(pf.particles.items()) if v > 0])) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(58, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations

    grader.require_is_equal(6, pf.particles[(3, 9)])
    grader.require_is_equal(0, pf.particles[(2, 10)])
    grader.require_is_equal(3, pf.particles[(8, 4)])
    grader.require_is_equal(2, pf.particles[(12, 6)])
    grader.require_is_equal(2, pf.particles[(7, 8)])
    grader.require_is_equal(2, pf.particles[(11, 6)])
    grader.require_is_equal(0, pf.particles[(18, 7)])
    grader.require_is_equal(1, pf.particles[(20, 5)])

    pf.elapseTime()
    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(57, len([k for k, v in list(pf.particles.items()) if v > 0])) # Slightly more particles lie on the same (row, col) locations

    grader.require_is_equal(4, pf.particles[(3, 9)])
    grader.require_is_equal(0, pf.particles[(2, 10)]) # 0 --> 0
    grader.require_is_equal(5, pf.particles[(8, 4)])
    grader.require_is_equal(3, pf.particles[(12, 6)])
    grader.require_is_equal(0, pf.particles[(7, 8)])
    grader.require_is_equal(2, pf.particles[(11, 6)])
    grader.require_is_equal(0, pf.particles[(18, 7)]) # 0 --> 1
    grader.require_is_equal(1, pf.particles[(20, 5)]) # 1 --> 0

grader.add_basic_part('3a-1-basic', test3a_1, 2, description="3a basic test for PF elapseTime")

def test3a_2():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.require_is_equal(69, len([k for k, v in list(pf.particles.items()) if v > 0])) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.require_is_equal(58, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations
    pf.observe(555, 193, 800)

    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(2, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations
    grader.require_is_equal(0.025, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 6))
    grader.require_is_equal(0.975, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(22, 7))

    pf.elapseTime()
    grader.require_is_equal(4, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations

    pf.observe(660, 193, 50)
    grader.require_is_equal(0.0, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(22, 6))
    grader.require_is_equal(1.0, pf.belief.getProb(22, 7))

grader.add_basic_part('3a-2-basic', test3a_2, 3, description="3a basic test for PF observe AND elapseTime")

def test3a_3i(): # basic observe stress test
    random.seed(34)
    A = 30
    B = 30
    pf = submission.ParticleFilter(A, B)

    N = 50
    p_values = []
    for i in range(N):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      a = int(random.random() * 30)
      b = int(random.random() * 5)
      c = int(random.random() * 30)

      random.seed(seed)
      pf.observe(a, b, c)
      random.seed(seed)
      for d in range(A):
        for e in range(B):
          p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)



grader.add_hidden_part('3a-3i-hidden', test3a_3i, 2, description="3a advanced test for PF observe")

def test3a_3ii(): # observe stress test with whether they put the pdf in the correct order or not
    random.seed(34)

    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # You can't swap a and c now!
      return a + b
    util.pdf = pdf

    A = 30
    B = 30
    random.seed(34)
    pf = submission.ParticleFilter(A, B)

    N = 50
    p_values = []
    for i in range(N):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      a = int(random.random() * 30)
      b = int(random.random() * 5)
      c = int(random.random() * 30)

      random.seed(seed)
      pf.observe(a, b, c)
      for d in range(A):
        for e in range(B):
          p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)


    util.pdf = oldpdf # fix the pdf

grader.add_hidden_part('3a-3ii-hidden', test3a_3ii, 2, description="3a test for pdf ordering")

def test3a_4():
    A = 30
    B = 30
    random.seed(35)
    pf = submission.ParticleFilter(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      random.seed(seed)
      pf.elapseTime()

      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)


grader.add_hidden_part('3a-4-hidden', test3a_4, 3, description="advanced test for PF elapseTime")

def test3a_5():
    A = 30
    B = 30
    random.seed(36)
    pf = submission.ParticleFilter(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      seed2 = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      random.seed(seed)
      pf.elapseTime()

      a = int(random.random() * 5 * A)
      b = int(random.random() * 5)
      c = int(random.random() * 5 * A)

      random.seed(seed2)
      pf.observe(a, b, c)
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)


grader.add_hidden_part('3a-5-hidden', test3a_5, 4, description="advanced test for PF observe AND elapseTime")

### Problem 4: which car is it?

grader.add_manual_part('4a', 5, description="conditional distribution")
grader.add_manual_part('4b', 4, description="number of assignments K!")
grader.add_manual_part('4c', 2, description="treewidth (extra credit)", extra_credit=True)
grader.add_manual_part('4d', 4, description="shifted car positions (extra credit)", extra_credit=True)


grader.add_manual_part('5a', 4, description='autonomous vehicles')
grader.add_manual_part('5b', 3, description='developing potentially lethal technology')

grader.grade()
