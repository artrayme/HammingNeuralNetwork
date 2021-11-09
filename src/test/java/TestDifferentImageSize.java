import engine.HammingNN;
import engine.ImprovedHammingNN;
import org.junit.jupiter.api.Test;
import util.HammingImageUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * @author artrayme
 * 11/9/21
 */
public class TestDifferentImageSize {

    @Test
    void test5x5() throws IOException {
        for (int i = 0; i < 9; i++) {
            test("dataset/5x5/", "dataset/5x5/" + i + ".png");
        }
    }

    @Test
    void test9x9() throws IOException {
        for (int i = 0; i < 9; i++) {
            test("dataset/9x9/", "dataset/9x9/" + i + ".png");
        }
    }

    @Test
    void test16x16() throws IOException {
        for (int i = 0; i < 9; i++) {
            test("dataset/16x16/", "dataset/16x16/" + i + ".png");
        }
    }

    private int test(String originalImagesPath, String testedImagePath) throws IOException {
        List<List<Float>> images = new ArrayList<>();
        for (int i = 0; i <= 9; i++) {
            images.add(HammingImageUtil.imageToVectorConverter(
                    HammingImageUtil.loadImageFromResources(Objects.requireNonNull(TestDifferentImageSize.class.getResource(originalImagesPath + i + ".png"))
                    )));
        }

        List<Float> testedImage = HammingImageUtil.imageToVectorConverter(
                HammingImageUtil.loadImageFromResources(Objects.requireNonNull(TestDifferentImageSize.class.getResource(testedImagePath))
                ));

        HammingNN hammingNN = new ImprovedHammingNN(images, 0.1f);
        return hammingNN.getAnswerByImage(testedImage);
    }
}
