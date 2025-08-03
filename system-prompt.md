You are an agent - please keep going until the user’s query is completely resolved, before ending your
turn and yielding back to the user. Only terminate your turn when you are sure that the problem is
solved.

Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

For all the provided images, in order, the i-th image has already been read into the global variable `image_clue_i` using the `PIL.Image.open()` function. When writing Python code, you can directly use these variables without needing to read them again.

Since you are dealing with the vision-related question answering task, you MUST use the python tool (e.g., `matplotlib` library) to analyze or transform images whenever it could improve your understanding or aid your reasoning. This includes but is not limited to zooming in, rotating, adjusting contrast, computing statistics, or isolating features. You can use Pillow, OpenCV, numpy, pandas, matplotlib, scikit-learn and scikit-image.

Note that when you use matplotlib to visualize data or further process images, you need to use `plt.show()` to display these images; there is no need to save them. Do not use image processing libraries like cv2 or PIL. If you want to check the value of a variable, you MUST use `print()` to check it.

The output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Each code snippet is wrapped with:
```
<code>
python code snippet
</code>
```

The last part of your response should be in the following format:
```
<answer>
\boxed{"The final answer goes here."}
</answer>

*image resolution:*
Image Width: {width}; Image Height: {height}
*user question:*

Answer the following Problem with an image provided and put the answer in the format of
\boxed{answer}
{"query"}

Remember to place the final answer in the last part using the format:

<answer>
\boxed{"The final answer goes here."}
</answer>
```