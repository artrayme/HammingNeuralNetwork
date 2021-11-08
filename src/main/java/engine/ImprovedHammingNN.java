package engine;

import model.DefaultFloatMatrix;
import model.FloatMatrix;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author artrayme
 * 11/9/21
 */
public class ImprovedHammingNN implements HammingNN {

    private final double maxError;

    private final FloatMatrix weights1;
    private final FloatMatrix weights2;

    public ImprovedHammingNN(List<List<Float>> images, double maxError) {
        this.maxError = maxError;
        weights1 = new DefaultFloatMatrix(images);
        weights1.scale(0.5f);
        weights2 = new DefaultFloatMatrix(images.size(), images.size());
        initWeights2();
    }

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
        FloatMatrix firstLayerOutput = weights1.mult(new DefaultFloatMatrix(image, true).transpose());
        var secondLayer = getSecondLayerResult(image, firstLayerOutput);
        System.out.println(secondLayer);

        return getIndexOfBestImage(secondLayer);
    }

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

    @Override
    public FloatMatrix getPatterns() {
        return weights1;
    }
}
