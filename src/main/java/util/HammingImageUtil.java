package util;

import engine.HammingNN;
import model.FloatMatrix;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author artrayme
 * 11/5/21
 */
public class HammingImageUtil {
    private static final float BLACK = 1.0f;

    private static final float WHITE = -1.0f;

    public static BufferedImage loadImageFromResources(URL path) throws IOException {
        return ImageIO.read(path);
    }

    public static List<Float> imageToVectorConverter(BufferedImage image) {
        Float[][] colors = new Float[image.getWidth()][image.getHeight()];
        for (int row = 0; row < image.getWidth(); row++) {
            for (int column = 0; column < image.getHeight(); column++) {
                colors[row][column] = defineColorValue(image.getRGB(row, column));
            }
        }
        return twoDArrayToList(colors);
    }

    private static <T> List<T> twoDArrayToList(T[][] twoDArray) {
        List<T> list = new ArrayList<>();
        for (T[] array : twoDArray) {
            list.addAll(Arrays.asList(array));
        }
        return list;
    }

    private static float defineColorValue(int rgb) {
        return new Color(rgb).equals(Color.BLACK) ? BLACK : WHITE;
    }

    public static void printAllCorrectImages(HammingNN hammingNN) {
        FloatMatrix weights1 = hammingNN.getPatterns();
        System.out.println("All correct IMAGES:");
        for (int indexImage = 0; indexImage < weights1.getHeight(); indexImage++) {
            System.out.println("Image №" + (indexImage));
            printOneImage(weights1, indexImage);
        }
    }

    public static void printOneImage(FloatMatrix weights1, int indexImage) {
        for (int indexPixel = 0; indexPixel < weights1.getWidth(); indexPixel++) {
            System.out.print(weights1.toArray()[indexImage][indexPixel] == 1 ? ".  " : "#  ");
            if ((indexPixel + 1) % 9 == 0) {
                System.out.println();
            }
        }
    }

}
