### Gradio

import gradio as gr
import torch
import numpy as np
from PIL import Image

# print(gr.__version__)

"""### Gradio Numbers"""

# def add_numbers(x,y):
#   return x + y

def sub_numbers(x,y):
  return x - y

# add_numbers(10,50)
sub_numbers(10,50)

# interface = gr.Interface(fn=add_numbers,
#                          inputs=[gr.Number(),gr.Number()],
#                          outputs=gr.Number())

interface = gr.Interface(fn=sub_numbers,
                         inputs=[gr.Number(),gr.Number()],
                         outputs=gr.Number())

interface.launch(share=True)

"""### Gradio Text"""

def reverse_text(text):
  return text[::-1]

reverse_text("Hello")

interface = gr.Interface(fn=reverse_text,
                         inputs=[gr.Text()],
                         outputs=gr.Text())

interface.launch(share=True)

"""### Gradio Sliders"""

def slider_example(value):
  return f"Slider current value is : {value}"

slider_example(10)

interface = gr.Interface(fn=slider_example,
                         inputs=gr.Slider(minimum=0,maximum=100,step=1),
                         outputs=gr.Text())

interface.launch(share=True)

"""### Gradio Dropdown Menu"""

def dropdown_example(value):
    return f"Current value is: {value}"

dropdown_example("Python")

options = ["Python", "PHP", "SQL"]

iface = gr.Interface(fn=dropdown_example,
                     inputs=gr.Dropdown(choices=options),
                     outputs=gr.Text())
iface.launch()

"""### Gradio Image"""

from PIL import Image

def convert(image_path):
  image = Image.open(image_path)
  return image.convert("L")

convert("/content/sea.jpg")

iface = gr.Interface(fn=convert,
                      inputs=gr.Image(type="filepath"),
                      outputs=gr.Image())

iface.launch(share=True)

"""### Gradio JSON"""

def number_details(number):
  details={
      "original":number,
      "squared": number ** 2,
      "sqrt": number ** 0.5,
      "is_even": number % 2 == 0,
  }

  return details

number_details(20)

iface = gr.Interface(fn=number_details,
                     inputs=gr.Number(),
                     outputs=gr.Json())
iface.launch(share=True)

"""### Gradio Label"""

def classify_number(number):
    if number > 0:
        return "positive"
    elif number < 0:
        return "negative"
    else:
        return "Zero"

classify_number(0)

iface = gr.Interface(fn=classify_number,
                     inputs=gr.Number(),
                     outputs=gr.Text())

iface.launch(share=True)

"""### Gradio Layout"""

with gr.Blocks() as demo:
  with gr.Row():
    text1 = gr.Text(value="OUTPUT ONE")
    text2 = gr.Text(value="OUTPUT TWO")

  with gr.Row():
    text3 = gr.Text(value="BOTTOM Row")

demo.launch(share=True)

with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column():
            text1 = gr.Text(value="Row 0 Col 0 - Comp 1")
            text2 = gr.Text(value="Row 0 Col 0 - Comp 2")

        with gr.Column():
            text3 = gr.Text(value="Row 0 Col 1")

    with gr.Row():
        text4 = gr.Text(value="Bottom row")

demo.launch()

with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=2): # 2/3
            text1 = gr.Text(value="Row 0 Col 0 - Comp 1")
            text2 = gr.Text(value="Row 0 Col 0 - Comp 2")

        with gr.Column(scale=1): # 1/3
            text3 = gr.Text(value="Row 0 Col 1")

    with gr.Row():
        text4 = gr.Text(value="Bottom Row")

demo.launch(share=True)

"""### Gradio Tabs"""

with gr.Blocks() as demo:

    with gr.Tab("Tab one"):

        with gr.Row():
            with gr.Column(scale=1):

                text1 = gr.Text(value="Row 0 Col 0 - Comp1")
                text2 = gr.Text(value="Row 0 Col 0 - Comp2")

            with gr.Column(scale=2):
                text3 = gr.Text(value="Row 0 Col 1")

        with gr.Row():
            text4 = gr.Text(value="Bottom row")

    with gr.Tab("Tab two"):
        with gr.Row():
            gr.Text("Welcome to the new tab!")

demo.launch(share=True)

"""### Gradio Accordion"""

with gr.Blocks() as demo:
    gr.Label("Label here")
    with gr.Accordion("Accordion here", open=True):
        gr.Image()

demo.launch()

"""### Gradio Buttons"""

with gr.Blocks() as demo:
    gr.Button("Button Comp one")
    gr.Button("Button Comp two")
    gr.Image()

demo.launch(share=True)

"""### Gradio CSS"""

css = """
.yourclass {
    height: 1000px;
    background-color: yellow;
}
"""

with gr.Blocks(css=css) as my_own_demo:
    with gr.Row(elem_classes=["yourclass"]):
        gr.Image(height=900, width=20)

my_own_demo.launch(share=True)

"""### Events -> Click Events , Change Events"""

def multiply(x,y):
  return x * y

multiply(10,6)

"""### Change Event"""

with gr.Blocks() as app:
  with gr.Row():
    x_slider = gr.Slider()
    y_slider = gr.Slider()

  with gr.Row():
    result = gr.Text()

  x_slider.change(fn=multiply,
                  inputs=[x_slider,y_slider],
                  outputs=result)

  y_slider.change(fn=multiply,
                  inputs=[x_slider,y_slider],
                  outputs=result)

  app.launch(share=True)

"""### Click Event"""

with gr.Blocks() as app:
  with gr.Row():
    x_slider = gr.Slider(label="X")
    y_slider = gr.Slider(label="Y")

  with gr.Row():
    result = gr.Text()

  with gr.Row():
    button = gr.Button("Multiply!")

  button.click(fn=multiply,
               inputs=[x_slider,y_slider],
               outputs=[result])

app.launch(share=True)

"""### Returning Different Outputs"""

def make_grayscale(image_path):
  image = Image.open(image_path)

  image_grayscale = image.convert("L")

  return image_grayscale , "Image converted"

# make_grayscale("/content/sea.jpg")

with gr.Blocks() as demo:
  with gr.Row():
    input_image = gr.Image(type="filepath")
    output_image = gr.Image()

  with gr.Row():
    log = gr.Text()
    submit = gr.Button(value="Convert to grayscale")

  submit.click(fn=make_grayscale,
               inputs=input_image,
               outputs=[output_image,log]
               )

demo.launch(share=True)

"""### Gradio Themes"""

# help(gr.themes)   # get theme list

def echo(num):
  return num

# echo(10)

iface = gr.Interface(fn=echo,
                     inputs=gr.Number(),
                     outputs=gr.Number(),
                     theme=gr.themes.Glass())

iface.launch(share=True)

"""### Gradio Integration with Machine Learning"""

from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

def classify_image(image):

  image = processor(image,return_tensors="pt")["pixel_values"]

  logits = model(image).logits

  predicted_label = logits.argmax(-1).item()

  return model.config.id2label[predicted_label]

iface = gr.Interface(fn=classify_image,
                     inputs=gr.Image(),
                     outputs=gr.Label(),
                     description="Upload an Image")

iface.launch(share=True)

"""### Gradio Sentiment Analysis"""

from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis")

def predict_sentiment(text):
  result = sentiment_analysis(text)
  return result[0]["label"] , result[0]["score"]

predict_sentiment("Today I am very delighted to have started this course")

iface = gr.Interface(fn=predict_sentiment,
                     inputs=gr.Textbox(lines=2 , placeholder="Type your text here"),
                     outputs=[gr.Text(label="Sentiment"),gr.Text(label="Score")])

iface.launch(share=True)

"""### Gradio Errors"""

def validate_even(num):
    return num % 2 == 0

# validate_even(11)

def add_even(a, b):
    if not validate_even(a):
        gr.Info("FIRST VALUE IS NOT EVEN!")
        # gr.Warning("FIRST VALUE IS NOT EVEN!")
        # gr.Error("FIRST VALUE IS NOT EVEN!")

    if not validate_even(b):
        gr.Info("SECOND VALUE IS NOT EVEN!")

    return a + b

iface = gr.Interface(fn=add_even,
                     inputs=[gr.Number(),gr.Number()],
                     outputs=gr.Number())

iface.launch(share=True)

def _combined_interface_function(context):
    import json
    from transformers import pipeline

    entity_json_string = find_entity(context)
    entities_list_of_dicts = json.loads(entity_json_string)

    # Better entity usage
    entity_descriptions = [
        f"{item['entity']} ({item['label']})"
        for item in entities_list_of_dicts
    ]

    sentiment_output = find_sentiment(context)

    generator = pipeline("text-generation", model="gpt2")

    data = {
        "entities": entity_descriptions,
        "sentiment": sentiment_output,
        "text": context
    }

    prompt = f"""
    Analyze the following user input and generate a meaningful summary.

    Sentiment: {data['sentiment']}
    Entities: {', '.join(data['entities'])}

    Message:
    {data['text']}

    Instructions:
    - Use entity roles correctly (person, place, organization)
    - Reflect the sentiment in tone
    - Keep it concise and logical

    Response:
    """

    output = generator(prompt, max_length=120, num_return_sequences=1)
    generated_text = output[0]["generated_text"]

    return generated_text, entities_list_of_dicts