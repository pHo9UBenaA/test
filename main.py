import os
import re
import torch
import werkzeug
from torchvision import transforms
from PIL import Image
from flask import Flask, request, redirect, session, url_for, render_template
from werkzeug.utils import secure_filename


title ="画風変換アプリケーション"
content_size = 512
img_url = '/var/www/app/static/trash/'
model_path = '/var/www/app/static/model/'

app = Flask(__name__, subdomain_matching=True)
app.secret_key = 'hogehoge'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
app.config['SERVER_NAME'] = '192.168.123.124:5000'

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

def stylize(content_image, content_size, model, cuda):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = load_image(content_image, size=content_size)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).to(device)
        output = output.cpu()
    return output[0]

def load_image(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        if img.size[0] > img.size[1]:
            img = img.resize((size, int(size*(img.size[1]/img.size[0]))), Image.NEAREST)
        else:
            img = img.resize((int(size*(img.size[0]/img.size[1])), size), Image.NEAREST)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def encode_base64(img_path):
    import base64
    with open(img_path, "rb") as f:
        bImg_base64 = base64.b64encode(f.read())
    strImg_base64 = str(bImg_base64)
    strImg_base64 = strImg_base64[2: len(strImg_base64) -1]
    base64 = "data:image/png;base64," + strImg_base64
    return base64


def allowed_file(filename):
    return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg'}

@app.route("/")
def index():
    return render_template("index.html", title=title)

@app.route('/post', methods=['GET', 'POST'])
def post():
    content_image = None
    if 'img_file' in request.files:
        content_image = request.files['img_file']
        pth_path = model_path + request.form['radio']

    if content_image and allowed_file(content_image.filename):
        filename = secure_filename(content_image.filename)
        img_path = os.path.join(img_url, filename)
        output = stylize(content_image=content_image, \
                           content_size=content_size, model=pth_path, cuda=1)
        save_image(img_path, output)
        session['img_path'] = img_path
    else:
        session['message'] = 'JPGまたはJPEGのファイルを選択してください。'

    return redirect(url_for('upload'))

@app.route('/upload')
def upload():
    img = None
    message = None
    if 'img_path' in session:
        img = encode_base64(session['img_path'])
        os.remove(session['img_path'])
    elif 'message' in session:
        message = session['message']
    session.clear()
    return render_template("index.html", title=title, result_img=img, message=message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
