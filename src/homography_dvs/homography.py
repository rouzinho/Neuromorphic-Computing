import numpy as np
import cv2

def get_points(namefile):
   file = open(namefile, "r")
   l = []
   while True:
      d = file.readline()
      if not d:
         break
      d = d.split()
      l.append(d)
   file.close()
   res = np.array(l).astype(int)
   
   return res

def get_size(namefile):
   file = open(namefile, "r")
   l = []
   while True:
      d = file.readline()
      if not d:
         break
      d = d.split()
      l.append(d)
   file.close()
   res = np.array(l).astype(int)
   min_w = 1000
   max_w = 0
   min_h = 1000
   max_h = 0
   print(res.shape)
   for i in res:
      if i[0] < min_w:
         min_w = i[0]
      if i[0] > max_w:
         max_w = i[0]
      if i[1] < min_h:
         min_h = i[1]
      if i[1] > max_h:
         max_h = i[1]
   print("min w ",min_w)
   print("max w ",max_w)
   print("min h ",min_h)
   print("max h ",max_h)
   width = max_w - min_w
   height = max_h - min_h

   return [width,height], [min_w,max_w,min_h,max_h]


if __name__ == "__main__":
   name_depth = "depth_points.txt"
   name_dvs = "points_dvs.txt"
   #get depth and dvs points
   depth_pts = get_points(name_depth)
   dvs_pts = get_points(name_dvs)
   size, borders = get_size(name_dvs)
   #compute homography
   h, status = cv2.findHomography(depth_pts, dvs_pts)
   #save homography to config files for ros kinect
   name_conf = "homography.yaml"
   line = "hom_depth_to_dvs: ["
   t = np.reshape(h,(9,1))
   params = ""
   for i in t:
      params = params + str(i[0]) + ", "
   params = params[:-2]
   line = line + params + "]\n"
   width = "width: " + str(size[0]) + "\n"
   height = "height: " + str(size[1]) + "\n"
   min_w = "min_w: " + str(borders[0]) + "\n"
   max_w = "max_w: " + str(borders[1]) + "\n"
   min_h = "min_h: " + str(borders[2]) + "\n"
   max_h = "max_h: " + str(borders[3]) + "\n"
   f = open(name_conf, "w")
   f.write(line)
   f.write(width)
   f.write(height)
   f.write(min_w)
   f.write(max_w)
   f.write(min_h)
   f.write(max_h)
   f.close()   
   
