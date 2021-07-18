import cv2

img=cv2.imread("Moon sinking, sun rising.jpg",1)


print(img)
newmoon=cv2.resize(img,(100,100))

cv2.imshow("Moon",newmoon)
cv2.waitKey(0)
cv2.imwrite("new_moon.jpg",newmoon)
cv2.destroyAllWindows()