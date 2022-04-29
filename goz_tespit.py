import cv2


ornekResim = cv2.imread("images/image.jpg", 1)


yuz_cascade = cv2.CascadeClassifier("cascade/yuz.xml")
goz_cascade = cv2.CascadeClassifier("cascade/goz.xml")

griTon = cv2.cvtColor(ornekResim, cv2.COLOR_BGR2GRAY)

yuz = yuz_cascade.detectMultiScale(griTon, 1.3, 2)
for (x, y, w, h) in yuz:
    cv2.rectangle(ornekResim, (x, y), (x+w, y+h), (0, 0, 255), 2)

    goz = goz_cascade.detectMultiScale(griTon[y:y+h, x:x+w], 1.3, 3)
    for (x1, y1, w1, h1) in goz:
        cv2.rectangle(ornekResim[y:y+h, x:x+w],
                      (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)


cv2.imshow("Gozler", ornekResim)
cv2.waitKey()
cv2.destroyAllWindows()
