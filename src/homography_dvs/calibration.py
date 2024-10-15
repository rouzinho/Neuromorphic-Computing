import fire
import os
import math
import json
import numpy as np
import cv2
from scipy.optimize import fsolve

from metavision_sdk_core import BaseFrameGenerationAlgorithm, RoiFilterAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.dat_tools import DatWriter
import metavision_sdk_cv

delta_t = 10000
nb_points_to_accumulate = 50

class PairOfBlinkingLedExtractor:
   """
   This class is used to detect a pair of blinking LEDs.

   The blinking frequencies (which must be different) are specified.
   Each processed chunk of events must contain exactly only one blob for each frequency, otherwise no clusters is returned
   """

   def __init__(self, height, width, min_freq=10., max_freq=510., min_cluster_size=3000, max_time_diff=delta_t,
               expected_low_freq=150., expected_high_freq=200.):
      self.height = height
      self.width = width

      self.frequency_filter = metavision_sdk_cv.FrequencyAlgorithm(
         width=width, height=height, min_freq=min_freq, max_freq=max_freq)
      self.frequency_clustering_filter = metavision_sdk_cv.FrequencyClusteringAlgorithm(
         width=width, height=height, min_cluster_size=min_cluster_size, max_time_diff=max_time_diff)
      self.freq_buffer = self.frequency_filter.get_empty_output_buffer()
      self.cluster_buffer = self.frequency_clustering_filter.get_empty_output_buffer()

      self.expected_low_freq = expected_low_freq
      self.expected_high_freq = expected_high_freq

   def compute_clusters(self, events):
      """
      Explicitly detects two clusters with specified frequencies
      """
      self.frequency_filter.process_events(events, self.freq_buffer)
      self.frequency_clustering_filter.process_events(self.freq_buffer, self.cluster_buffer)
      if self.cluster_buffer.numpy().size == 1:
         cluster_buffer_np = self.cluster_buffer.numpy().copy()

         return cluster_buffer_np
      else:
         return []
      
def calibration(input_recording):
   #cv2.namedWindow('events',cv2.WINDOW_NORMAL)
   mv_it = EventsIterator(input_path=input_recording, delta_t=delta_t)
   ev_height, ev_width = mv_it.get_size()
   events_frame = np.zeros((ev_height, ev_width, 3), dtype=np.uint8)
   blinking_leds_extractor = PairOfBlinkingLedExtractor(height=ev_height, width=ev_width)
   sum_x = 0
   sum_y = 0
   count = 0
   color = (255, 0, 0)
   for ev in mv_it:
      if ev.size == 0:
         continue

      BaseFrameGenerationAlgorithm.generate_frame(ev, events_frame)
      clusters = blinking_leds_extractor.compute_clusters(ev)
      for cluster in clusters:
         count+=1
         #print("x : ",int(cluster["x"]))
         #print("y : ",int(cluster["y"]))
         sum_x = sum_x + int(cluster["x"])
         sum_y = sum_y + int(cluster["y"])
         x0 = int(cluster["x"]) - 10
         y0 = int(cluster["y"]) - 10
         cv2.rectangle(events_frame, (x0, y0), (x0 + 20, y0 + 20), color=(0, 0, 255))
         cv2.putText(events_frame, "{} Hz".format(int(cluster["frequency"])), (x0, y0 - 10), cv2.FONT_HERSHEY_PLAIN,
                     1, (0, 0, 255), 1)

      
      
      cv2.imshow('events', events_frame[..., ::-1])
      key = cv2.waitKey(1)
      if count >= 300:
         break
   mean_x = int(sum_x / count)
   mean_y = int(sum_y / count)
   print("mean_x : ",mean_x)
   print("mean_y : ",mean_y)

   return mean_x, mean_y

if __name__ == "__main__":
   #filename = "/home/altair/Postdoc/Codes/homography_dvs/calib_raw.raw"
   filename = ""
   m_x, m_y = calibration(filename)
   f = open("points_dvs.txt", "a")
   line = str(m_x) + " " + str(m_y) + "\n"
   f.write(line)