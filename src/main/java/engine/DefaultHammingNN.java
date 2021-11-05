package engine;

/**
 * @author artrayme
 * 11/5/21
 */
public class DefaultHammingNN implements HammingNN {

    private final int imageHeight;
    private final int imageWidth;
    private final double maxError;

    DefaultHammingNN(int imageHeight, int imageWidth, double maxError) {
        this.imageHeight = imageHeight;
        this.imageWidth = imageWidth;
        this.maxError = maxError;
    }


    @Override
    public void learnImage(boolean[][] image, int answer) {

    }

    @Override
    public int getAnswerByImage(boolean[][] image) {
        return 0;
    }


}
