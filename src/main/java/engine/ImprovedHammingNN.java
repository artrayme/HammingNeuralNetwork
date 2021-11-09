package engine;

import model.DefaultFloatMatrix;
import model.FloatMatrix;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author artrayme
 * 11/5/21
 * @see <a href="https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%A5%D1%8D%D0%BC%D0%BC%D0%B8%D0%BD%D0%B3%D0%B0">Wikipedia article about Hamming</a>
 * @see <a href="https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050917X00021/1-s2.0-S1877050917301278/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHXxFByLDL%2BnG0%2FwxooM5BGrMx%2BvEInn9tvNE9S%2BCMMZAiEAhp5q7OqW7u0MXvGGIysq8djpWA%2F4DbOydwl5eurnLvIqgwQIpP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDLbiCdOpyHrq64r2xyrXA6ibPE90X%2Fo9K4%2FXwKH1VEEsUMRhoUY6p8fp%2FHV%2Bz1S2NnhHWObNVbXvBRzhfeHIOMsSbkJvJPyacwY0m1CxpnHaGNo1bYsyZv18%2B%2BNpvNFdiLTJWNpRo7bWwke7egDXddLxdj0%2B9mFGrCklPK3VYbhOLeF3ncw85qngt2a0jILUPF7YuINRowE263ld%2FCTM36bMMfPREwK25lsFLk8fe9VsP8Vddxs9786LA%2FI%2FlRhPvt6cBX%2FH8Ho5o7qzi4j%2BnZfwe2fzkNQP0QhtdTYoUTOSfyxxw%2BOXl2uZNyEAf6ljAZQkLqtlFmBkt2%2BySiJgxgp0Ul2mH9bSI6IXuG5DZvJ1IBNNgVWZNBWw4MjI%2FV6Ht7QejHnsHz2FeN7Obz%2BPUYdpGwTnfnDj0uxyJKXV3%2F31EV2cmi5hA7NsU52ac4MjQxSidT3NM0btUpPc3cj%2FB9Wbyegy5LoLHHrUDfyuzl0E8kqaPBEeD4YPztcEGyoorpCiPIJAWaFVAfRzjHhTRiIsP5%2BUXWUOrQp7jnq4QS70PxAabDKMtHZqnbaKxHPsxCoWQk%2FC17Mctyq4wMS1slXmutUejejuNLSkkVD0MuX4MetbwII4moBiZqhj4ciVZ1csHqMX%2BjDs6p6MBjqlAQjeuYjjm8sJAmQutWPb41aJ33ebnQ6Sdsgafht6SiF40qndHVj0WWOuVuzBOHaPuiyyIHlrQMkt0eFFw%2BxXvkCdrcctCQW7IGW02c74uelQF7Jt3iN8s4yDrTaKzQg4cjECPC%2Fq42f%2BzO%2FJdKgDjZGRdjINKAysPRvCyOyVbteG0XxPdnOA1fwWwxFZLvPDQB1AUK5tTY9ze328rYi5Yi3nrk0vxg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211107T121053Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2SIEF3N7%2F20211107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0a3264b7992b15bcda2a5a65c67c6e2e06c8a2b0efb092c29c13e66a78301db7&hash=d1b9bc704e1f9772809f482c0be49b12613594153e63bf719bee18cb5bf6751b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050917301278&tid=spdf-47fb0800-3b2d-485b-bf4a-6ef9f9336995&sid=0abd5c0c525b3741f629cf12cf097682f4ccgxrqb&type=client">Article about object classification using Hamming Neural Network</a>
 */
public class ImprovedHammingNN implements HammingNN {

    /**
     * Max error value. Neural network must resolve image until
     * the difference between the two iteration is less than this value
     * <p></p>
     * The normal value is 0.1
     */
    private final double maxError;

    /**
     * First layer weights matrix.
     * This matrix is initialized when creating the network
     * and does not change during work
     * <p></p>
     * width = pixels count in one image
     * height = learned images count
     */
    private final FloatMatrix weights1;

    /**
     * Second layer weights matrix.
     * This matrix is initialized when creating the network
     * and does not change during work.
     * But every cycle of image recognition a new vector comes in
     * <p></p>
     * width = height = learned images count
     */
    private final FloatMatrix weights2;

    /**
     * Main constructor.
     * You must pass in constructor list of input images vectors.
     * Each vector must be the same length as the others.
     * You can't add new well-known image to the trained network.
     * <p></p>
     * Also, you must pass the max error. A Typical value is 0.1.
     *
     * @param images   - List of will-known images. Each image is a List of
     *                 values {-1, 1}
     * @param maxError - value of max recognition error. Typically, is 0.1
     */
    public ImprovedHammingNN(List<List<Float>> images, double maxError) {
        this.maxError = maxError;
        weights1 = new DefaultFloatMatrix(images);
        weights1.scale(0.5f);
        weights2 = new DefaultFloatMatrix(images.size(), images.size());
        initWeights2();
    }

    /**
     * Initialisation weight at second layer.
     * The matrix will be of the form.
     * <pre>
     * {@code
     * {
     *     {1, -e, -e, .... -e}
     *     {-e, 1, -e, .... -e}
     *     {-e, -e, 1, .... -e}
     *     {.., .., .., .., ..}
     *     {-e, -e, -e, ...  1}
     * }
     * }
     * </pre>
     * Say, K = well-known images count (equal matrix width and height)<br>
     * Then {@code 0<e<1/K}
     */

    private void initWeights2() {
        float eps = (float) 1 / weights2.getWidth();
        for (int i = 0; i < weights2.getWidth(); i++) {
            for (int j = 0; j < weights2.getWidth(); j++) {
                if (i == j) {
                    weights2.toArray()[i][j] = 1;
                } else {
                    weights2.toArray()[i][j] = -ThreadLocalRandom.current().nextFloat(0, eps);
                }
            }
        }
    }

    @Override
    public int getAnswerByImage(List<Float> image) {
        if (image.size() != weights1.getWidth())
            throw new IllegalArgumentException("Passed image size = " + image.size()
                    + ". But this NN can only work with images of size" + weights1.getWidth());
        FloatMatrix firstLayerOutput = weights1.mult(new DefaultFloatMatrix(image, true).transpose());
        var secondLayer = getSecondLayerResult(image, firstLayerOutput);
        System.out.println(secondLayer);

        return getIndexOfBestImage(secondLayer);
    }

    @Override
    public FloatMatrix getPatterns() {
        return weights1;
    }

    /**
     * This method finds max value in the state of second layer
     *
     * @return index of max value
     */
    private int getIndexOfBestImage(FloatMatrix secondLayer) {
        int indexOfBestImage = 0;
        float lastMax = 0f;
        for (int j = 0; j < secondLayer.getHeight(); j++) {
            if (secondLayer.toArray()[j][0] > lastMax) {
                lastMax = secondLayer.toArray()[j][0];
                indexOfBestImage = j;
            }
        }
        return indexOfBestImage;
    }

    /**
     * Main cycle of image recognizing.
     * In the first iteration the input is the output of the first layer.
     * After each iteration of the image recognition,
     * the error is calculated as the sum of the
     * difference between the past image and the obtained.
     *
     * @return last second layer state
     */
    private FloatMatrix getSecondLayerResult(List<Float> image, FloatMatrix secondLayer) {
        double currentError = 0;
        do {
            FloatMatrix newMatrix = weights2.mult(secondLayer);
            currentError = secondLayer.minus(newMatrix).sum() / image.size();
            secondLayer = newMatrix;
            System.out.println("Current error = " + currentError);
        } while (currentError > maxError);
        return secondLayer;
    }
}
