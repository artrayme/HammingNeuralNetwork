import engine.DefaultHammingNN;
import engine.HammingImageUtil;
import engine.HammingNN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * @author artrayme
 * 11/5/21
 */
public class Main {


    public static void main(String[] args) throws IOException {
        List<List<Float>> images = new ArrayList<>();
        for (int i = 0; i <= 9; i++) {
            images.add(HammingImageUtil.imageToVectorConverter(
                    HammingImageUtil.loadImageFromResources(Objects.requireNonNull(Main.class.getResource("dataset/9x9/" + i + ".png"))
                    )));
        }

        List<Float> badImage = HammingImageUtil.imageToVectorConverter(
                HammingImageUtil.loadImageFromResources(Objects.requireNonNull(Main.class.getResource("9x9/" + 1 + ".png"))
                ));

        List<Float> goodImage = HammingImageUtil.imageToVectorConverter(
                HammingImageUtil.loadImageFromResources(Objects.requireNonNull(Main.class.getResource("dataset/9x9/" + 4 + ".png"))
                ));

        HammingNN hammingNN = new DefaultHammingNN(images, 0.1);
        //        hammingNN.printAllCorrectImages();
        var res = hammingNN.getAnswerByImage(badImage);
        System.out.println("Image number " + res);
//        HammingImageUtil.printAllCorrectImages(hammingNN);
    }

}


