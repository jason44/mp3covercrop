# mp3covercrop

There are two methods for framing, v1 and v2/v3. v2 is generally better at framing around subjects (especially non-centered ones), but there are some regressions. In particular, v2 performs worst for images where the foreground is centered, but the background is high density. v1 overcomes this particular problem by penalizing frames that move away from the center, which is not ideal because this results in poor framing for some images with non-centered foreground.

v3 is generally better than v1 and v2 at framing subjects. v3 also adds a --center flag to force a frame to center. Overall, v2 and v3 may require more manual intervention for images that need to be centered, but they produce better results overall. 

