package engine;

import model.DefaultFloatMatrix;
import model.FloatMatrix;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * links:
 * https://www.youtube.com/watch?v=W7ux1RfOQeM&ab_channel=Kirsanov2011
 * <p></p>
 * https://www.youtube.com/watch?v=Fe1UFkLbAx4&ab_channel=RomanShamin
 * <p></p>
 * http://www.codenet.ru/progr/alg/ai/htm/gl3_5.php
 * <p></p>
 * https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%A5%D1%8D%D0%BC%D0%BC%D0%B8%D0%BD%D0%B3%D0%B0
 * <p></p>
 * https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050917X00021/1-s2.0-S1877050917301278/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHXxFByLDL%2BnG0%2FwxooM5BGrMx%2BvEInn9tvNE9S%2BCMMZAiEAhp5q7OqW7u0MXvGGIysq8djpWA%2F4DbOydwl5eurnLvIqgwQIpP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDLbiCdOpyHrq64r2xyrXA6ibPE90X%2Fo9K4%2FXwKH1VEEsUMRhoUY6p8fp%2FHV%2Bz1S2NnhHWObNVbXvBRzhfeHIOMsSbkJvJPyacwY0m1CxpnHaGNo1bYsyZv18%2B%2BNpvNFdiLTJWNpRo7bWwke7egDXddLxdj0%2B9mFGrCklPK3VYbhOLeF3ncw85qngt2a0jILUPF7YuINRowE263ld%2FCTM36bMMfPREwK25lsFLk8fe9VsP8Vddxs9786LA%2FI%2FlRhPvt6cBX%2FH8Ho5o7qzi4j%2BnZfwe2fzkNQP0QhtdTYoUTOSfyxxw%2BOXl2uZNyEAf6ljAZQkLqtlFmBkt2%2BySiJgxgp0Ul2mH9bSI6IXuG5DZvJ1IBNNgVWZNBWw4MjI%2FV6Ht7QejHnsHz2FeN7Obz%2BPUYdpGwTnfnDj0uxyJKXV3%2F31EV2cmi5hA7NsU52ac4MjQxSidT3NM0btUpPc3cj%2FB9Wbyegy5LoLHHrUDfyuzl0E8kqaPBEeD4YPztcEGyoorpCiPIJAWaFVAfRzjHhTRiIsP5%2BUXWUOrQp7jnq4QS70PxAabDKMtHZqnbaKxHPsxCoWQk%2FC17Mctyq4wMS1slXmutUejejuNLSkkVD0MuX4MetbwII4moBiZqhj4ciVZ1csHqMX%2BjDs6p6MBjqlAQjeuYjjm8sJAmQutWPb41aJ33ebnQ6Sdsgafht6SiF40qndHVj0WWOuVuzBOHaPuiyyIHlrQMkt0eFFw%2BxXvkCdrcctCQW7IGW02c74uelQF7Jt3iN8s4yDrTaKzQg4cjECPC%2Fq42f%2BzO%2FJdKgDjZGRdjINKAysPRvCyOyVbteG0XxPdnOA1fwWwxFZLvPDQB1AUK5tTY9ze328rYi5Yi3nrk0vxg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211107T121053Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2SIEF3N7%2F20211107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0a3264b7992b15bcda2a5a65c67c6e2e06c8a2b0efb092c29c13e66a78301db7&hash=d1b9bc704e1f9772809f482c0be49b12613594153e63bf719bee18cb5bf6751b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050917301278&tid=spdf-47fb0800-3b2d-485b-bf4a-6ef9f9336995&sid=0abd5c0c525b3741f629cf12cf097682f4ccgxrqb&type=client
 *
 * @author artrayme
 * 11/5/21
 */
public class DefaultHammingNN implements HammingNN {

    private final double maxError;

    /**
     * T_k = n/2, k=0..m-1;
     * But activation threshold same for all images
     * cause images has same length (n)
     */
    private final float activationThreshold;


    private final FloatMatrix weights1;
    private final FloatMatrix weights2;

    public DefaultHammingNN(List<List<Float>> images, double maxError) {
        this.maxError = maxError;
        weights1 = new DefaultFloatMatrix(images);
        weights1.scale(0.5f);
        weights2 = new DefaultFloatMatrix(images.get(0).size(), images.get(0).size());
        activationThreshold = (float) images.get(0).size() * 2;
        initWeights2();
    }

    private void initWeights2() {
        float eps = (float) 1 / weights2.getWidth();
        for (int i = 0; i < weights2.getWidth(); i++) {
            for (int j = 0; j < weights2.getWidth(); j++) {
                if (i == j) {
                    weights2.toArray()[i][j] = 1;
                } else {
                    weights2.toArray()[i][j] = ThreadLocalRandom.current().nextFloat(0, eps);
                }
            }
        }
    }

    private float activation(float value) {
        value += 81;
        if (value <= 0) {
            return 0f;
        } else if (value > activationThreshold) {
            return activationThreshold;
        }
        return value-81;
    }

    @Override
    public int getAnswerByImage(List<Float> image) {
        FloatMatrix firstLayerOutput = weights1.mult(new DefaultFloatMatrix(image, true).transpose());
        FloatMatrix secondLayer = calcNewSecondLayerState(firstLayerOutput);

        double delta;
        do {
            var lastValue = secondLayer;
            secondLayer = calcNewSecondLayerState(calcNewSecondLayerAxons(secondLayer));
            delta = lastValue.minus(secondLayer).sum() / image.size();
            System.out.println("Current delta = " + delta);
        } while (delta > maxError);

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

    @Override
    public FloatMatrix getPatterns() {
        return weights1;
    }

    private FloatMatrix calcNewSecondLayerState(FloatMatrix lastState) {
        FloatMatrix result = new DefaultFloatMatrix(1, lastState.getHeight());
        for (int i = 0; i < result.getHeight(); i++) {
            float sum = 0;
            for (int j = 0; j < weights2.getWidth(); j++) {
                sum += weights2.toArray()[i][j];
            }
            result.toArray()[i][0] = lastState.toArray()[i][0] - sum;
        }
        return result;
    }

    private FloatMatrix calcNewSecondLayerAxons(FloatMatrix secondLayerState) {
        FloatMatrix result = new DefaultFloatMatrix(1, secondLayerState.getHeight());
        for (int i = 0; i < secondLayerState.getHeight(); i++) {
            result.toArray()[i][0] = activation(secondLayerState.toArray()[i][0]);
//                        result.toArray()[i][0] = (secondLayerState.toArray()[i][0]);
        }
        return result;
    }

    private FloatMatrix calcFirstLayerState(List<Float> image) {
        FloatMatrix result = new DefaultFloatMatrix(1, image.size());
        for (int i = 0; i < image.size(); i++) {
            float sum = 0;
            for (int j = 0; j < weights1.getHeight(); j++) {
                sum += weights1.toArray()[j][i];
            }
            result.toArray()[i][0] = sum + activationThreshold;
        }
        return result;
    }
}
