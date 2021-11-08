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

    /**
     * T_k = n/2, k=0..m-1;
     * But activation threshold same for all images
     * cause images has same length (n)
     */
    private final float activationThreshold;


    private final FloatMatrix weights1;
    private final FloatMatrix weights2;

    public ImprovedHammingNN(List<List<Float>> images, double maxError) {
        this.maxError = maxError;
        weights1 = new DefaultFloatMatrix(images);
        weights1.scale(0.5f);
        weights2 = new DefaultFloatMatrix(images.size(), images.size());
        activationThreshold = (float) images.get(0).size() /2;
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
        FloatMatrix secondLayer = firstLayerOutput;
        double currentError = 0;
        do {
            FloatMatrix newMatrix = weights2.mult(secondLayer);
            currentError = secondLayer.minus(newMatrix).sum()/image.size();
            secondLayer = newMatrix;
            System.out.println("Current error = " + currentError);
        }while (currentError>maxError );
        System.out.println(secondLayer);

        int indexOfBestImage = 0;
        float lastMax = 0f;
        for (int j = 0; j < secondLayer.getHeight(); j++) {
            if (secondLayer.toArray()[j][0] > lastMax) {
                lastMax = secondLayer.toArray()[j][0];
                indexOfBestImage = j;
            }
        }
        System.err.println(indexOfBestImage);
        return indexOfBestImage;


    }

    @Override
    public FloatMatrix getPatterns() {
        return null;
    }
}
