import cv2

coord_x = 0
coord_y = 0


def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        print(f"mouse click: ({x}, {y})")

        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)

img = cv2.imread('boardpic.jpg')  

cv2.imshow('image', img)
cv2.setMouseCallback('image', show_coordinates)

while True:
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
