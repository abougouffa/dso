#pragma once

#include <stdio.h>
#include <iostream>

#include <glog/logging.h>
#include <boost/thread.hpp>

#include "util/num_type.h"
#include "util/settings.h"

namespace dso {

template <typename Running>
class IndexThreadReduce {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  inline IndexThreadReduce() {
    nextIndex = 0;
    maxIndex = 0;
    stepSize = 1;
    callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault, this,
                               _1, _2, _3, _4);

    running = true;
    for (int i = 0; i < NUM_THREADS; ++i) {
      isDone[i] = false;
      gotOne[i] = true;
      workerThreads[i] = boost::thread(&IndexThreadReduce::workerLoop, this, i);
    }
  }
  inline ~IndexThreadReduce() {
    running = false;

    exMutex.lock();
    todo_signal.notify_all();
    exMutex.unlock();

    for (int i = 0; i < NUM_THREADS; ++i) {
      workerThreads[i].join();
    }

    printf("destroyed ThreadReduce\n");
  }

  inline void reduce(
      boost::function<void(int, int, Running*, int)> callPerIndex, int first,
      int end, int stepSize = 0) {
    memset(&stats, 0, sizeof(Running));

    //		if(!multiThreading)
    //		{
    //			callPerIndex(first, end, &stats, 0);
    //			return;
    //		}

    if (stepSize == 0) {
      stepSize = ((end - first) + NUM_THREADS - 1) / NUM_THREADS;
    }

    // printf("reduce called\n");

    boost::unique_lock<boost::mutex> lock(exMutex);

    // save
    this->callPerIndex = callPerIndex;
    nextIndex = first;
    maxIndex = end;
    this->stepSize = stepSize;

    // go worker threads!
    for (int i = 0; i < NUM_THREADS; ++i) {
      isDone[i] = false;
      gotOne[i] = false;
    }

    // let them start!
    todo_signal.notify_all();

    // printf("reduce waiting for threads to finish\n");
    // wait for all worker threads to signal they are done.
    while (true) {
      // wait for at least one to finish
      done_signal.wait(lock);
      // printf("thread finished!\n");

      // check if actually all are finished.
      bool allDone = true;
      for (int i = 0; i < NUM_THREADS; ++i) {
        allDone = allDone && isDone[i];
      }

      // all are finished! exit.
      if (allDone) {
        break;
      }
    }

    nextIndex = 0;
    maxIndex = 0;
    this->callPerIndex = boost::bind(&IndexThreadReduce::callPerIndexDefault,
                                     this, _1, _2, _3, _4);

    // printf("reduce done (all threads finished)\n");
  }

  Running stats;

 private:
  boost::thread workerThreads[NUM_THREADS];
  bool isDone[NUM_THREADS];
  bool gotOne[NUM_THREADS];

  boost::mutex exMutex;
  boost::condition_variable todo_signal;
  boost::condition_variable done_signal;

  int nextIndex;
  int maxIndex;
  int stepSize;

  bool running;

  boost::function<void(int, int, Running*, int)> callPerIndex;

  void callPerIndexDefault(int i, int j, Running* k, int tid) {
    LOG(ERROR) << "ERROR: should never be called....";
    assert(false);
  }

  void workerLoop(int idx) {
    boost::unique_lock<boost::mutex> lock(exMutex);

    while (running) {
      // try to get something to do.
      int todo = 0;
      bool gotSomething = false;
      if (nextIndex < maxIndex) {
        // got something!
        todo = nextIndex;
        nextIndex += stepSize;
        gotSomething = true;
      }

      // if got something: do it (unlock in the meantime)
      if (gotSomething) {
        lock.unlock();

        assert(callPerIndex != 0);

        Running s;
        memset(&s, 0, sizeof(Running));
        callPerIndex(todo, std::min(todo + stepSize, maxIndex), &s, idx);
        gotOne[idx] = true;
        lock.lock();
        stats += s;
      }

      // otherwise wait on signal, releasing lock in the meantime.
      else {
        if (!gotOne[idx]) {
          lock.unlock();
          assert(callPerIndex != 0);
          Running s;
          memset(&s, 0, sizeof(Running));
          callPerIndex(0, 0, &s, idx);
          gotOne[idx] = true;
          lock.lock();
          stats += s;
        }
        isDone[idx] = true;
        // printf("worker %d waiting..\n", idx);
        done_signal.notify_all();
        todo_signal.wait(lock);
      }
    }
  }
};
}
