import numpy
import time
import tensorflow as tf
import holisticModule as hm
import prepareData as prep
import numpy as np
import cv2

PATH_MODEL = 'C:\\Users\\DougC\\Desktop\\Special\\RNN_Final-151-0.75.model'
PATH_VIDEO = 'C:\\Users\\DougC\\Desktop\\Special\\variado_V1-0007.mp4'


CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']


def main():
    new_model = tf.keras.models.load_model(PATH_MODEL)
    detector = hm.HolisticDetector()
    prepare = prep.Preprocessor()
    cap = cv2.VideoCapture(PATH_VIDEO)  # PATH_VIDEO or CAMERA_FEED
    ret, frame = cap.read()
    pastTime = 0
    vec = np.zeros((84, 1))
    while ret:
        ret, frame = cap.read()
        if not ret:
            print('FLAG')
            break
        detector.find_pose(frame)
        detector.draw_pose(frame)
        detector.get_lm()
        mat1, mat2, mat3 = detector.get_matrix()
        feat_vec = prepare.get_distMat(mat1, mat2, mat3)
        feat_vec = np.array(feat_vec)
        mag = np.true_divide(feat_vec[:, 0], 2202)  # normalize dist for 1920x1080
        ang = np.true_divide(feat_vec[:, 1], 360)  # normalize ang for 360
        aux = numpy.concatenate((mag, ang))
        vec = np.column_stack((vec, aux))
        currentTime = time.time()
        fps = int(1 / (currentTime - pastTime))
        pastTime = currentTime
        cv2.putText(frame, f'FPS:{fps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("original", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    print(vec.shape)
    while vec.shape[1] < 136:  # 136*2
        vec = np.column_stack((vec, np.zeros(84)))
    print(vec.shape)
    if vec.shape[1] == 136:
        print("ENTRO")
        print(type(vec))
        vec = np.asarray(vec, dtype= np.float32).reshape(-1, 136, 84)
        prediction = new_model.predict(vec, steps=1, verbose=0)
        print(CATEGORY[np.argmax(prediction)])
        # print(f"Video corresponde a {new_model.predict(vec)}")
        # model.predict(x=valid_x, steps=len(valid_y), verbose=0)
        # vec = np.zeros(84)


if __name__ == '__main__':
    main()
