package engine;

/**
 * @author artrayme
 * 11/5/21
 */
public interface HammingNN {

    void learnImage(boolean[][] image, int answer);

    int getAnswerByImage(boolean[][] image);
}
