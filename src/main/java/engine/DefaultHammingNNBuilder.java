package engine;

public class DefaultHammingNNBuilder {
    private int imageHeight;
    private int imageWidth;
    private double maxError;

    public DefaultHammingNNBuilder setImageHeight(int imageHeight) {
        this.imageHeight = imageHeight;
        return this;
    }

    public DefaultHammingNNBuilder setImageWidth(int imageWidth) {
        this.imageWidth = imageWidth;
        return this;
    }

    public DefaultHammingNNBuilder setMaxError(double maxError) {
        this.maxError = maxError;
        return this;
    }

    public DefaultHammingNN createDefaultHammingNN() {
        return new DefaultHammingNN(imageHeight, imageWidth, maxError);
    }
}