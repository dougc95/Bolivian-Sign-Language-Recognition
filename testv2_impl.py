import numpy
import time
import tensorflow as tf
import holisticModule as hm
import prepareData as prep
import numpy as np
import cv2


# PATH_MODEL = 'C:\\Users\\DougC\\Desktop\\Special\\RNN_Final-151-0.75.model'
# PATH_MODEL = "C:\\Users\\DougC\\Desktop\\Special\\model88LSTM\\models\\BiLSTMCuDNN0.4-196-0.88.model"
PATH_MODEL = "C:\\Users\\DougC\\Desktop\\Special\\test88.h5"
PATH_VIDEO = 'C:\\Users\\DougC\\Desktop\\Special\\variado_V1-0007.mp4'
# PATH_VIDEO = 'C:\\Users\\DougC\\Desktop\\LSBv1\\Donde\\donde_V1-0001.mp4'


CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']


def main():
    detector = hm.HolisticDetector()
    prepare = prep.Preprocessor()
    new_model = tf.keras.models.load_model(PATH_MODEL)

    cap = cv2.VideoCapture(PATH_VIDEO)  # PATH_VIDEO or CAMERA_FEED
    pastTime = 0

    vec = np.zeros((84, 1))
    frame_counter = 0
    seq_length = 75

    ret, frame = cap.read()
    while ret:
        print(frame_counter)
        ret, frame = cap.read()
        if not ret:
            print('FLAG')
            while vec.shape[1] < seq_length:
                vec = np.column_stack((vec, np.zeros(84)))
            print("ENTRO")
            print(type(vec))
            vec = np.asarray(vec, dtype=np.float32).reshape(-1, seq_length, 84)
            prediction = new_model.predict(vec, steps=1, verbose=0)
            inference = CATEGORY[np.argmax(prediction)]
            print(inference)
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
        frame_counter += 1
        vec = np.column_stack((vec, aux))
        # new_mat = np.delete(vec, -1, axis=1)
        # new_mat = np.insert(new_mat, 0, aux, axis=1)
        # temp = np.asarray(new_mat, dtype= np.float32).reshape(-1, seq_length, 84)
        currentTime = time.time()
        fps = int(1 / (currentTime - pastTime))
        pastTime = currentTime
        cv2.putText(frame, f'FPS:{fps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("original", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        if frame_counter < seq_length:
            continue
        else:
            frame_counter = 0
        vec = np.delete(vec, -1, axis=1)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1, seq_length, 84)
        prediction = new_model.predict(vec, steps=1, verbose=0)
        inference = CATEGORY[np.argmax(prediction)]
        print(inference)
        vec = np.zeros((84, 1))


if __name__ == '__main__':
    main()
