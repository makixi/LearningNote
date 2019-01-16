# 1.Getting Started

### 1.1 first step

```html 
<html lang="en">
<head>
  <script src="https://cesiumjs.org/releases/1.53/Build/Cesium/Cesium.js"></script>
  <style>
      @import url(../Build/Cesium/Widgets/widgets.css);
      html, body, #cesiumContainer {
          width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
      }
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <script>
    Cesium.Ion.defaultAccessToken = your_asset_tokens;
    var viewer = new Cesium.Viewer('cesiumContainer');
  </script>
</body>
</html>
```

---

### 1.2 Adding Cesium World Terrain
```javascript
var viewer = new Cesium.Viewer('cesiumContainer', {
    terrainProvider: Cesium.createWorldTerrain()
});
```

运行时在Terrain选择Cesium World Terrain模式

在Cesium Ion上传kmp文件，得到AssetToken号，在地图上放置模型，并默认运行时缩放到模型处
PS:需要指定自己的<code>Cesium.Ion.defaultAccessToken</code>。


```javascript
var tileset = viewer.scene.primitives.add(
    new Cesium.Cesium3DTileset({
        url: Cesium.IonResource.fromAssetId(asset_id)
    })
);
viewer.zoomTo(tileset);
```

---

### 1.3 the Viewer 
![1](https://cesiumjs.org/tutorials/Cesium-Workshop/images/viewerAnnotated.jpg)<br>
1.Geocoder<br>
2.HomeButton<br>
3.SceneModePicker<br>
4.BaseLayerPicker<br>
5.NavigationHelpButton<br>
6.Animation<br>
7.CreditsDisplay<br>
8.Timeline<br>
9.FullscreenButton

so,以下这样可以创建一个没有选择指示器、底层选择器或场景模式选择器小部件的查看器

```javascript
var viewer = new Cesium.Viewer('cesiumContainer', {
    //terrainProvider: Cesium.createWorldTerrain(),
    scene3DOnly: true,
    selectionIndicator: false,
    baseLayerPicker: false
});
```

PS:显示帧速(FPS)
```javascript
viewer.scene.debugShowFramesPerSecond=true;
```

---

### 1.4 Adding Imagery
#### 1.4.1 Adding basic imagery
首先viewer处可以添加图片，这里添加的**网络**图片

```javascript
var viewer = new Cesium.Viewer('cesiumContainer', {
    //添加图片
    imageryProvider:new Cesium.ArcGisMapServerImageryProvider({
        url:'https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer'
    })
});
```

可以通过**Ion**添加图片<br>
PS:可以调整透明度和亮度

```javascript
var earthAtNight = viewer.imageryLayers.addImageryProvider(new Cesium.IonImageryProvider({ assetId: 3812 }));
earthAtNight.alpha = 0.5;//透明度
earthAtNight.brightness = 10.0;//亮度
```

通过**本地**添加图片,PS:不加限定即全部伸展覆盖

```javascript
viewer.imageryLayers.addImageryProvider(new Cesium.SingleTileImageryProvider({
    url : '../Apps/Sandcastle/images/Cesium_Logo_overlay.png',
    rectangle : Cesium.Rectangle.fromDegrees(-75.0, 28.0, -67.0, 29.75)
}));

```

#### 1.4.2 Adjusting imagery colors
```html 
<style>
    @import url(../templates/bucket.css);
    #toolbar {
        background: rgba(42, 42, 42, 0.8);
        padding: 4px;
        border-radius: 4px;
    }
    #toolbar input {
        vertical-align: middle;
        padding-top: 2px;
        padding-bottom: 2px;
    }
</style>
<div id="cesiumContainer" class="fullSize"></div>
<div id="loadingOverlay"><h1>Loading...</h1></div>
<div id="toolbar">
<table>
    <tbody><tr>
        <td>Brightness</td>
        <td>
            <input type="range" min="0" max="3" step="0.02" data-bind="value: brightness, valueUpdate: 'input'">
            <input type="text" size="5" data-bind="value: brightness">
        </td>
    </tr>
    <tr>
        <td>Contrast</td>
        <td>
            <input type="range" min="0" max="3" step="0.02" data-bind="value: contrast, valueUpdate: 'input'">
            <input type="text" size="5" data-bind="value: contrast">
        </td>
    </tr>
    <tr>
        <td>Hue</td>
        <td>
            <input type="range" min="0" max="3" step="0.02" data-bind="value: hue, valueUpdate: 'input'">
            <input type="text" size="5" data-bind="value: hue">
        </td>
    </tr>
    <tr>
        <td>Saturation</td>
        <td>
            <input type="range" min="0" max="3" step="0.02" data-bind="value: saturation, valueUpdate: 'input'">
            <input type="text" size="5" data-bind="value: saturation">
        </td>
    </tr>
    <tr>
        <td>Gamma</td>
        <td>
            <input type="range" min="0" max="3" step="0.02" data-bind="value: gamma, valueUpdate: 'input'">
            <input type="text" size="5" data-bind="value: gamma">
        </td>
    </tr>
</tbody></table>
</div>
```

```javascript
Cesium.Ion.defaultAccessToken = your_asset_tokens;

var viewer=new Cesium.Viewer('cesiumContainer');

var imageryLayers = viewer.imageryLayers;

//应用颜色状态
var viewModel = {
    brightness: 0,//亮度
    contrast: 0,//对比度
    hue: 0,//色相
    saturation: 0,//饱和度
    gamma: 0//透明度
};
//将viewmodel转换为knockout可见项
Cesium.knockout.track(viewModel);

//绑定viewmodel到UI(toolbar)
var toolbar = document.getElementById('toolbar');
Cesium.knockout.applyBindings(viewModel, toolbar);

// 监听控件的变化事件
function subscribeLayerParameter(name) {
    Cesium.knockout.getObservable(viewModel, name).subscribe(
        function(newValue) {
            if (imageryLayers.length > 0) {
                var layer = imageryLayers.get(0);
                layer[name] = newValue;
            }
        }
    );
}

subscribeLayerParameter('brightness');
subscribeLayerParameter('contrast');
subscribeLayerParameter('hue');
subscribeLayerParameter('saturation');
subscribeLayerParameter('gamma');

//值与控件绑定达到双向更新的目的
function updateViewModel() {
    if (imageryLayers.length > 0) {
        var layer = imageryLayers.get(0);
        viewModel.brightness = layer.brightness;
        viewModel.contrast = layer.contrast;
        viewModel.hue = layer.hue;
        viewModel.saturation = layer.saturation;
        viewModel.gamma = layer.gamma;
    }
}

imageryLayers.layerAdded.addEventListener(updateViewModel);
imageryLayers.layerRemoved.addEventListener(updateViewModel);
imageryLayers.layerMoved.addEventListener(updateViewModel);
updateViewModel();

```

---

#### 1.4.3 Manipulating and ordering imagery layers
```html 
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Use correct character set. -->
  <meta charset="utf-8">
  <!-- Tell IE to use the latest, best version. -->
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <!-- Make the application on mobile take up the full browser screen and disable user scaling. -->
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, user-scalable=no">
  <title>Hello World!</title>
  <script src="../Build/Cesium/Cesium.js"></script>
  <style>
      @import url(../Build/Cesium/Widgets/widgets.css);
      html, body, #cesiumContainer {
          width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
      }
  </style>
  <style>
      @import url(../Apps/Sandcastle/templates/bucket.css);
      #toolbar {
        background: rgba(42, 42, 42, 0.8);
        padding: 4px;
        border-radius: 4px;
      }
      #toolbar input {
        vertical-align: middle;
        padding-top: 2px;
        padding-bottom: 2px;
      }
      #toolbar table tr {
        transform: translateY(0);
        transition: transform 0.4s ease-out;
      }
      #toolbar table tr.up {
        transform: translateY(33px);
        transition: none;
      }
      #toolbar table tr.down {
        transform: translateY(-33px);
        transition: none;
      }
  </style>
</head>
<body>
  <div id="cesiumContainer" class="fullSize"></div>
  <div id="loadingOverlay"><h1>Loading...</h1></div>
  <div id="toolbar">
    <table>
      <tbody data-bind="foreach: layers">
        <tr data-bind="css: { up: $parent.upLayer === $data, down: $parent.downLayer === $data }">
          <td><input type="checkbox" data-bind="checked: show"></td>
          <td>
            <span data-bind="text: name, visible: !$parent.isSelectableLayer($data)"></span>
            <select data-bind="visible: $parent.isSelectableLayer($data), options: $parent.baseLayers, optionsText: 'name', value: $parent.selectedLayer"></select>
          </td>
          <td>
            <input type="range" min="0" max="1" step="0.01" data-bind="value: alpha, valueUpdate: 'input'">
          </td>
          <td><button type="button" class="cesium-button" data-bind="click: function() { $parent.raise($data, $index()); }, visible: $parent.canRaise($index())">▲</button></td>
          <td><button type="button" class="cesium-button" data-bind="click: function() { $parent.lower($data, $index()); }, visible: $parent.canLower($index())">▼</button></td>
        </tr>
      </tbody>
    </table>
  </div>

  <script src="../Build/Cesium/trans.js"></script>
</body>
</html>

```

```javascript

Cesium.Ion.defaultAccessToken = your_asset_tokens;

var viewer=new Cesium.Viewer('cesiumContainer',{
    baseLayerPicker:false
});

var imageryLayers = viewer.imageryLayers;

var viewModel = {
    layers : [],
    baseLayers : [],
    upLayer : null,
    downLayer : null,
    selectedLayer : null,
    isSelectableLayer : function(layer) {
        return this.baseLayers.indexOf(layer) >= 0;
    },
    raise : function(layer, index) {
        imageryLayers.raise(layer);
        viewModel.upLayer = layer;
        viewModel.downLayer = viewModel.layers[Math.max(0, index - 1)];
        updateLayerList();
        window.setTimeout(function() { viewModel.upLayer = viewModel.downLayer = null; }, 10);
    },
    lower : function(layer, index) {
        imageryLayers.lower(layer);
        viewModel.upLayer = viewModel.layers[Math.min(viewModel.layers.length - 1, index + 1)];
        viewModel.downLayer = layer;
        updateLayerList();
        window.setTimeout(function() { viewModel.upLayer = viewModel.downLayer = null; }, 10);
    },
    canRaise : function(layerIndex) {
        return layerIndex > 0;
    },
    canLower : function(layerIndex) {
        return layerIndex >= 0 && layerIndex < imageryLayers.length - 1;
    }
};

var baseLayers = viewModel.baseLayers;

Cesium.knockout.track(viewModel);

function setupLayers() {
    // Create all the base layers that this example will support.
    // These base layers aren't really special.  It's possible to have multiple of them
    // enabled at once, just like the other layers, but it doesn't make much sense because
    // all of these layers cover the entire globe and are opaque.
    addBaseLayerOption(
            'Bing Maps Aerial',
            undefined); // the current base layer
    addBaseLayerOption(
            'Bing Maps Road',
            new Cesium.BingMapsImageryProvider({
                url : 'https://dev.virtualearth.net',
                mapStyle: Cesium.BingMapsStyle.ROAD
            }));
    addBaseLayerOption(
            'ArcGIS World Street Maps',
            new Cesium.ArcGisMapServerImageryProvider({
                url : 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer'
            }));
    addBaseLayerOption(
            'OpenStreetMaps',
            Cesium.createOpenStreetMapImageryProvider());
    addBaseLayerOption(
            'MapQuest OpenStreetMaps',
            Cesium.createOpenStreetMapImageryProvider({
                url : 'https://otile1-s.mqcdn.com/tiles/1.0.0/osm/'
            }));
    addBaseLayerOption(
            'Stamen Maps',
            Cesium.createOpenStreetMapImageryProvider({
                url : 'https://stamen-tiles.a.ssl.fastly.net/watercolor/',
                fileExtension: 'jpg',
                credit: 'Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under CC BY SA.'
            }));
    addBaseLayerOption(
            'Natural Earth II (local)',
            Cesium.createTileMapServiceImageryProvider({
                url : Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII')
            }));
    addBaseLayerOption(
            'USGS Shaded Relief (via WMTS)',
            new Cesium.WebMapTileServiceImageryProvider({
                url : 'http://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/WMTS',
                layer : 'USGSShadedReliefOnly',
                style : 'default',
                format : 'image/jpeg',
                tileMatrixSetID : 'default028mm',
                maximumLevel: 19,
                credit : 'U. S. Geological Survey'
            }));

    // Create the additional layers
    addAdditionalLayerOption(
            'United States GOES Infrared',
            new Cesium.WebMapServiceImageryProvider({
                url : 'https://mesonet.agron.iastate.edu/cgi-bin/wms/goes/conus_ir.cgi?',
                layers : 'goes_conus_ir',
                credit : 'Infrared data courtesy Iowa Environmental Mesonet',
                parameters : {
                    transparent : 'true',
                    format : 'image/png'
                }
            }));
    addAdditionalLayerOption(
            'United States Weather Radar',
            new Cesium.WebMapServiceImageryProvider({
                url : 'https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r.cgi?',
                layers : 'nexrad-n0r',
                credit : 'Radar data courtesy Iowa Environmental Mesonet',
                parameters : {
                    transparent : 'true',
                    format : 'image/png'
                }
            }));
    addAdditionalLayerOption(
            'TileMapService Image',
            Cesium.createTileMapServiceImageryProvider({
                url : '../Apps/Sandcastle/images/cesium_maptiler/Cesium_Logo_Color'
            }),
            0.2);
    addAdditionalLayerOption(
            'Single Image',
            new Cesium.SingleTileImageryProvider({
                url : '../Apps/Sandcastle/images/Cesium_Logo_overlay.png',
                rectangle : Cesium.Rectangle.fromDegrees(-115.0, 38.0, -107, 39.75)
            }),
            1.0);
    addAdditionalLayerOption(
            'Grid',
            new Cesium.GridImageryProvider(), 1.0, false);
    addAdditionalLayerOption(
            'Tile Coordinates',
            new Cesium.TileCoordinatesImageryProvider(), 1.0, false);
}

function addBaseLayerOption(name, imageryProvider) {
    var layer;
    if (typeof imageryProvider === 'undefined') {
        layer = imageryLayers.get(0);
        viewModel.selectedLayer = layer;
    } else {
        layer = new Cesium.ImageryLayer(imageryProvider);
    }

    layer.name = name;
    baseLayers.push(layer);
}

function addAdditionalLayerOption(name, imageryProvider, alpha, show) {
    var layer = imageryLayers.addImageryProvider(imageryProvider);
    layer.alpha = Cesium.defaultValue(alpha, 0.5);
    layer.show = Cesium.defaultValue(show, true);
    layer.name = name;
    Cesium.knockout.track(layer, ['alpha', 'show', 'name']);
}

function updateLayerList() {
    var numLayers = imageryLayers.length;
    viewModel.layers.splice(0, viewModel.layers.length);
    for (var i = numLayers - 1; i >= 0; --i) {
        viewModel.layers.push(imageryLayers.get(i));
    }
}

setupLayers();
updateLayerList();

//Bind the viewModel to the DOM elements of the UI that call for it.
var toolbar = document.getElementById('toolbar');
Cesium.knockout.applyBindings(viewModel, toolbar);

Cesium.knockout.getObservable(viewModel, 'selectedLayer').subscribe(function(baseLayer) {
    // Handle changes to the drop-down base layer selector.
    var activeLayerIndex = 0;
    var numLayers = viewModel.layers.length;
    for (var i = 0; i < numLayers; ++i) {
        if (viewModel.isSelectableLayer(viewModel.layers[i])) {
            activeLayerIndex = i;
            break;
        }
    }
    var activeLayer = viewModel.layers[activeLayerIndex];
    var show = activeLayer.show;
    var alpha = activeLayer.alpha;
    imageryLayers.remove(activeLayer, false);
    imageryLayers.add(baseLayer, numLayers - activeLayerIndex - 1);
    baseLayer.show = show;
    baseLayer.alpha = alpha;
    updateLayerList();
});

```

#### 1.4.4 Splitting imagery layers 
```html 
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Use correct character set. -->
  <meta charset="utf-8">
  <!-- Tell IE to use the latest, best version. -->
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <!-- Make the application on mobile take up the full browser screen and disable user scaling. -->
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, user-scalable=no">
  <title>Hello World!</title>
  <script src="../Build/Cesium/Cesium.js"></script>
  <style>
      @import url(../Build/Cesium/Widgets/widgets.css);
      html, body, #cesiumContainer {
          width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden;
      }
  </style>
  <style>
      @import url(../Apps/Sandcastle/templates/bucket.css);
      #slider {
        position: absolute;
        left: 50%;
        top: 0px;
        background-color: #D3D3D3;
        width: 5px;
        height: 100%;
        z-index: 9999;
      }

      #slider:hover {
        cursor: ew-resize;
      }
  </style>
</head>
<body>
  <div id="cesiumContainer" class="fullSize">
    <div id="slider"></div>
  </div>
  <div id="loadingOverlay"><h1>Loading...</h1></div>
  <div id="toolbar"></div>

  <script src="../Build/Cesium/trans.js"></script>
</body>
</html>

```

```javascript
Cesium.Ion.defaultAccessToken = your_asset_tokens;

var viewer=new Cesium.Viewer('cesiumContainer',{
    imageryProvider : new Cesium.ArcGisMapServerImageryProvider({
        url : 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer'
    }),
    baseLayerPicker : false,
    infoBox : false
});

var layers = viewer.imageryLayers;
var earthAtNight = layers.addImageryProvider(new Cesium.IonImageryProvider({ assetId: 3812 }));
earthAtNight.splitDirection = Cesium.ImagerySplitDirection.LEFT; // Only show to the left of the slider.

// // 将滑块位置与拆分位置同步
var slider = document.getElementById('slider');
viewer.scene.imagerySplitPosition = (slider.offsetLeft) / slider.parentElement.offsetWidth;//设置图层分割的位置 可以直接用0.5表示中间

var handler = new Cesium.ScreenSpaceEventHandler(slider);

var moveActive = false;

function move(movement) {
    if(!moveActive) {
        return;
    }

    var relativeOffset = movement.endPosition.x ;
    var splitPosition = (slider.offsetLeft + relativeOffset) / slider.parentElement.offsetWidth;
    slider.style.left = 100.0 * splitPosition + '%';
    viewer.scene.imagerySplitPosition = splitPosition;
}

handler.setInputAction(function() {
    moveActive = true;
}, Cesium.ScreenSpaceEventType.LEFT_DOWN);
handler.setInputAction(function() {
    moveActive = true;
}, Cesium.ScreenSpaceEventType.PINCH_START);

handler.setInputAction(move, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
handler.setInputAction(move, Cesium.ScreenSpaceEventType.PINCH_MOVE);

handler.setInputAction(function() {
    moveActive = false;
}, Cesium.ScreenSpaceEventType.LEFT_UP);
handler.setInputAction(function() {
    moveActive = false;
}, Cesium.ScreenSpaceEventType.PINCH_END);
```

---

### 1.5 Adding Terrain 
```javascript
var viewer=new Cesium.Viewer('cesiumContainer');

viewer.terrainProvider=Cesium.createWorldTerrain({
    requestWaterMask:true, //需要水效应
    requestVertexNormals:true //地形照明需要
});

//启用深度测试，使地形后面的东西消失
viewer.scene.globe.depthTestAgainstTerrain=true;
```

---

### 1.6 Configuring the Scene
```javascript
//根据太阳/月亮位置启用照明
viewer.scene.globe.enableLighting=true;
```

基本cesium类型：<br>
1.Cartesian3:一个三维笛卡尔坐标。使用地球固定框架相对于地球中心<br>
2.Cartographic:由经度、纬度（弧度）和高度（距离WGS84椭球面）定义的位置<br>
3.HeadingPitchRoll:Heading是围绕负z轴的旋转，pitch是围绕负y轴的旋转，roll是围绕正x轴的旋转。<br>
4.Quaternion:以4d坐标表示的三维旋转。

#### 1.6.1 Camera Control
一些最常用的方法是：<br>
- camera.setview（options）：立即将camera设置在特定位置和方向<br>
- camera.zoomin（amount）：沿视图向量向前移动camera<br>
- camera.zoomout（amount）：沿视图向量向后移动camera<br>
- camera.flyto（options）：创建从当前相机位置到新位置的动画相机飞行。<br>
- camera.look at（target，offset）：确定摄像机的方向和位置，以给定偏移量查看目标点。<br>
- camera.move（direction，amount）：将相机向任何方向移动<br>
- camera.rotate（axis，angle）：围绕任何轴旋转相机

设置相机初始位置与点击homebutton后flyto的位置

```javascript
// Create an initial camera view
var initialPosition = new Cesium.Cartesian3.fromDegrees(-73.998114468289017509, 40.674512895646692812, 2631.082799425431);
var initialOrientation = new Cesium.HeadingPitchRoll.fromDegrees(7.1077496389876024807, -31.987223091598949054, 0.025883251314954971306);
var homeCameraView = {
    destination : initialPosition,
    orientation : {
        heading : initialOrientation.heading,
        pitch : initialOrientation.pitch,
        roll : initialOrientation.roll
    }
};
// Set the initial view
viewer.scene.camera.setView(homeCameraView);

// Add some camera flight animation options
homeCameraView.duration = 2.0;
homeCameraView.maximumHeight = 2000;
homeCameraView.pitchAdjustHeight = 2000;
homeCameraView.endTransform = Cesium.Matrix4.IDENTITY;
// Override the default home button
viewer.homeButton.viewModel.command.beforeExecute.addEventListener(function (e) {
    e.cancel = true;
    viewer.scene.camera.flyTo(homeCameraView);
});
```

#### 1.6.2 Clock Control
设置场景时间选项

```javascript
viewer.clock.shouldAnimate = true; // make the animation play when the viewer starts
viewer.clock.startTime = Cesium.JulianDate.fromIso8601("2017-07-11T16:00:00Z");
viewer.clock.stopTime = Cesium.JulianDate.fromIso8601("2017-07-12T12:20:00Z");
viewer.clock.currentTime = Cesium.JulianDate.fromIso8601("2017-07-11T16:00:00Z");
viewer.clock.multiplier = 2; // sets a speedup
viewer.clock.clockStep = Cesium.ClockStep.SYSTEM_CLOCK_MULTIPLIER; // tick computation mode
viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP; // loop at the end
viewer.timeline.zoomTo(viewer.clock.startTime, viewer.clock.stopTime); // set visible range
```

```javascript
// Create a clock that loops on Christmas day 2013 and runs in 4000x real time.
var clock = new Cesium.Clock({
   startTime : Cesium.JulianDate.fromIso8601('2013-12-25'),
   currentTime : Cesium.JulianDate.fromIso8601('2013-12-25'),
   stopTime : Cesium.JulianDate.fromIso8601('2013-12-26'),
   clockRange : Cesium.ClockRange.LOOP_STOP, // loop when we hit the end time
   clockStep : Cesium.ClockStep.SYSTEM_CLOCK_MULTIPLIER,
   multiplier : 4000, // how much time to advance each tick
   shouldAnimate : true // Animation on by default
});

var viewer = new Cesium.Viewer('cesiumContainer', {
    clockViewModel : new Cesium.ClockViewModel(clock)
});

viewer.scene.globe.enableLighting = true;

Sandcastle.addToolbarButton('Reset Current Time', function() {
    var resetTime = viewer.clockViewModel.startTime;
    viewer.clockViewModel.currentTime = resetTime;
    viewer.timeline.updateFromClock();
});

Sandcastle.addToolbarButton('Slow Down Clock', function() {
    viewer.clockViewModel.multiplier /= 2;
});

Sandcastle.addToolbarButton('Speed Up Clock', function() {
    viewer.clockViewModel.multiplier *= 2;
});
```

---

### 1.7 Loading and Styling Entities
#### 1.7.1 polygon and polyline:
##### 1.7.1.1 Box
```javascript
var blueBox=viewer.entities.add({
    name:'Blue Box',
    position:Cesium.Cartesian3.fromDegrees(-114.0,40.0,300000.0),
    box:{
        dimensions:new Cesium.Cartesian3(400000.0,300000.0,500000.0),
        fill:true,
        material:Cesium.Color.BLUE.withAlpha(0.5),
        //new Cesium.ImageMaterialProperty({image:'xxx',color:Cesium.Color.BLUE,repeat:new Cesium.Cartesian2(4,4),transparent:false});
        //new Cesium.CheckedboardMaterialProperty({evenColor:Cesium.Color.WHITE,oddColor:Cesium.Color.BLACK,repeat:new Cesium.Cartesian2(4,4)});
        //new Cesium.StripeMaterialProperty({evenColor:Cesium.Color.WHITE,oddColor:Cesium.Color.BLACK,repeat:32,offset:20,orientation:Cesium.StripeOrientation.VERTICAL});
        //new Cesium.GridMaterialProperty({color:Cesium.Color.WHITE,cellAlpha:0.2,lineCount:new Cesium.Cartesian2(8,8),lineThickness:new Cesium.Cartesian2(2.0,2.0),lineOffset:new Cesium.Cartesian2(2.0,2.0)});

        outline:true,
        outlineColor:Cesium.Color.BLACK
    }
});
```

##### 1.7.1.2 Circles and Ellipses
```javascript
var Circle=viewer.entities.add({
    name:'Green circle at height with outline',
    position:Cesium.Cartesian3.fromDegrees(-114.0,40.0,300000.0),
    ellipse:{
        semiMinorAxis:300000.0,
        semiMajorAxis:250000.0,
        height:200000.0,
        fill:true,
        material:Cesium.Color.GREEN.withAlpha(0.5),
        outline:false,
        outlineColor:Cesium.Color.BLACK
    }
});

var Ellipse=viewer.entities.add({
    name:'Blue translucent, rotated, and extruded ellipse with outline',
    position:Cesium.Cartesian3.fromDegrees(-104.0,40.0,300000.0),
    ellipse:{
        semiMinorAxis:300000.0,
        semiMajorAxis:250000.0,
        extrudedHeight : 200000.0,
        rotation:Cesium.Math.toRadians(45),
        fill:true,
        material:Cesium.Color.BLUE.withAlpha(0.5),
        outline:true,
        outlineColor:Cesium.Color.BLACK
    }
});

```

##### 1.7.1.3 Corridor
```javascript

var Corridor=viewer.entities.add({
    name:'Red corridor',
    corridor:{
        positions:Cesium.Cartesian3.fromDegreesArray([
            -100.0,40.0,
            -105.0,40.0,
            -105.0,35.0,
            -110.0,30.0
        ]),
        height:200000.0,
        width:200000.0,
        extrudedHeight : 100000.0,
        cornerType:Cesium.CornerType.BEVELED,//Cesium.CornerType.MITERED
        fill:true,
        material:Cesium.Color.GREEN.withAlpha(0.5),
        outline:true,
        outlineColor:Cesium.Color.BLACK
    }
});
```

##### 1.7.1.4 Cylinders and Cones 
```javascript
var Cone=viewer.entities.add({
    name:'Cone',
    position:Cesium.Cartesian3.fromDegrees(-105.0,40.0,200000.0),
    cylinder:{
        topRadius:200000.0,
        bottomRadius:400000.0,
        length:400000.0,
        material:Cesium.Color.GREEN.withAlpha(0.5),
        outline:true,
        outlineColor:Cesium.Color.DARK_GREEN
    }
});
```

##### 1.7.1.5 Geometry Height Reference 
```javascript
var cesiumTerrainProvider = Cesium.createWorldTerrain();
var ellipsoidTerrainProvider = new Cesium.EllipsoidTerrainProvider();

var viewer = new Cesium.Viewer('cesiumContainer', {
    baseLayerPicker : false,
    terrainProvider : cesiumTerrainProvider
});

// depth test against terrain is required to make the polygons clamp to terrain
// instead of showing through it from underground
viewer.scene.globe.depthTestAgainstTerrain = true;

Sandcastle.addToolbarMenu([{
    text : 'Polygons',
    onselect : function() {
        viewer.entities.removeAll();
        addPolygons();
    }
}, {
    text : 'Boxes, Cylinders and Ellipsoids',
    onselect : function() {
        viewer.entities.removeAll();
        addGeometries();
    }
}]);

Sandcastle.addToolbarMenu([{
    text : 'Terrain Enabled',
    onselect : function() {
        viewer.scene.terrainProvider = cesiumTerrainProvider;
    }
}, {
    text : 'Terrain Disabled',
    onselect : function() {
        viewer.scene.terrainProvider = ellipsoidTerrainProvider;
    }
}]);

var longitude = 6.950615989890521;
var latitude = 45.79546589994886;
var delta = 0.001;

function addGeometry(i, j) {
    var west = longitude + delta * i;
    var north = latitude + delta * j + delta;

    var type = Math.floor(Math.random() * 3);
    if (type === 0) {
        viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(west, north, 0.0),
            box : {
                dimensions : new Cesium.Cartesian3(40.0, 30.0, 50.0),
                material : Cesium.Color.fromRandom({alpha : 1.0}),
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
            }
        });
    } else if (type === 1) {
        viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(west, north, 0.0),
            cylinder : {
                length :50.0,
                topRadius : 20.0,
                bottomRadius : 20.0,
                material : Cesium.Color.fromRandom({alpha : 1.0}),
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
            }
        });
    } else {
        viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(west, north, 0.0),
            ellipsoid : {
                radii : new Cesium.Cartesian3(20.0, 15.0, 25.0),
                material : Cesium.Color.fromRandom({alpha : 1.0}),
                heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
            }
        });
    }
}

function addGeometries(){
    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            addGeometry(i, j);
        }
    }
    viewer.zoomTo(viewer.entities);
}

function addPolygon(i, j) {
    var west = longitude + delta * i;
    var east = longitude + delta * i + delta;

    var south = latitude + delta * j;
    var north = latitude + delta * j + delta;
    var a = Cesium.Cartesian3.fromDegrees(west, south);
    var b = Cesium.Cartesian3.fromDegrees(west, north);
    var c = Cesium.Cartesian3.fromDegrees(east, north);
    var d = Cesium.Cartesian3.fromDegrees(east, south);

    var positions = [a, b, c, d];
    viewer.entities.add({
        polygon : {
            hierarchy : positions,
            material : Cesium.Color.fromRandom({alpha : 1}),
            height : 40.0,
            heightReference : Cesium.HeightReference.RELATIVE_TO_GROUND,
            extrudedHeight : 0.0,
            extrudedHeightReference : Cesium.HeightReference.CLAMP_TO_GROUND
        }
    });
}

function addPolygons() {
    // create 16 polygons that are side-by-side
    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 4; j++) {
            addPolygon(i, j);
        }
    }
    //viewer.zoomTo(viewer.entities);
    viewer.camera.lookAt(Cesium.Cartesian3.fromDegrees(longitude, latitude, 1500), new Cesium.HeadingPitchRange(-Cesium.Math.PI/2, -Cesium.Math.PI_OVER_FOUR, 2000));
    viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
}

```

##### 1.7.1.6 Plane
```javascript
var Plane=viewer.entities.add({
    name:'Plane',
    position:Cesium.Cartesian3.fromDegrees(-114.0, 40.0, 300000.0),
    plane : {
        plane : new Cesium.Plane(Cesium.Cartesian3.UNIT_X, 0.0),
        dimensions : new Cesium.Cartesian2(400000.0, 300000.0),
        fill:true,
        material : Cesium.Color.BLUE.withAlpha(0.5),
        ouline:true,
        outlineColor:Cesium.Color.BLACK
    }
});
```

##### 1.7.1.7 Polygon 
```javascript
var redPolygon = viewer.entities.add({
    name : 'Red polygon on surface',
    polygon : {
        hierarchy : Cesium.Cartesian3.fromDegreesArray([-115.0, 37.0,
                                                        -115.0, 32.0,
                                                        -107.0, 33.0,
                                                        -102.0, 31.0,
                                                        -102.0, 35.0]),
        material : Cesium.Color.RED,
        extrudedHeight: 500000.0,
        closeTop : true,
        closeBottom : true,
        outline : true,
        perPositionHeight : true,
        outlineColor : Cesium.Color.BLACK
    }
});

var bluePolygon = viewer.entities.add({
    name : 'Blue polygon with holes',
    polygon : {
        hierarchy : {
            positions : Cesium.Cartesian3.fromDegreesArray([-99.0, 30.0,
                                                            -85.0, 30.0,
                                                            -85.0, 40.0,
                                                            -99.0, 40.0]),
            holes : [{
                positions : Cesium.Cartesian3.fromDegreesArray([
                    -97.0, 31.0,
                    -97.0, 39.0,
                    -87.0, 39.0,
                    -87.0, 31.0
                ]),
                holes : [{
                    positions : Cesium.Cartesian3.fromDegreesArray([
                        -95.0, 33.0,
                        -89.0, 33.0,
                        -89.0, 37.0,
                        -95.0, 37.0
                    ]),
                    holes : [{
                        positions : Cesium.Cartesian3.fromDegreesArray([
                            -93.0, 34.0,
                            -91.0, 34.0,
                            -91.0, 36.0,
                            -93.0, 36.0
                        ])
                    }]
                }]
            }]
        },
        material : Cesium.Color.BLUE.withAlpha(0.5),
        height : 0,
        outline : true // height is required for outline to display
    }
});
```

#### 1.7.1.8 Polyline Dash 
```javascript
var redLine = viewer.entities.add({
    name : 'Red dashed line',
    polyline : {
        positions : Cesium.Cartesian3.fromDegreesArrayHeights([-75, 38, 250000,
                                                        -125, 38, 250000]),
        width : 5,
        material : new Cesium.PolylineDashMaterialProperty({
            color: Cesium.Color.RED,
            gapColor: Cesium.Color.YELLOW,
            dashLength: 8.0,
            dashPattern: parseInt('110000001111', 2)
        })
    }
});
```

##### 1.7.1.9  Polyline Volume 
```javascript
var greenBox = viewer.entities.add({
    name : 'Green box with beveled corners and outline',
    polylineVolume : {
        positions : Cesium.Cartesian3.fromDegreesArrayHeights([-90.0, 32.0, 0.0,
                                                               -90.0, 36.0, 100000.0,
                                                               -94.0, 36.0, 0.0]),
        shape :[new Cesium.Cartesian2(-50000, -50000),
                new Cesium.Cartesian2(50000, -50000),
                new Cesium.Cartesian2(50000, 50000),
                new Cesium.Cartesian2(-50000, 50000)],//computeStar(7, 70000, 50000) computeCircle(60000.0)
        cornerType : Cesium.CornerType.BEVELED,//Cesium.CornerType.MITERED
        material : Cesium.Color.GREEN.withAlpha(0.5),
        outline : true,
        outlineColor : Cesium.Color.BLACK
    }
});
```

##### 1.7.1.10 Polyline 
```javascript 
var redLine = viewer.entities.add({
    name : 'Red line on terrain',
    polyline : {
        positions : Cesium.Cartesian3.fromDegreesArray([-75, 35,
                                                        -125, 35]),
        width : 5,
        followSurface : true,
        material : Cesium.Color.RED,
        //new Cesium.PolylineGlowMaterialProperty({glowPower : 0.2,color : Cesium.Color.BLUE})
        //new Cesium.PolylineOutlineMaterialProperty({color : Cesium.Color.ORANGE,outlineWidth : 2,outlineColor : Cesium.Color.BLACK})
        //new Cesium.PolylineArrowMaterialProperty(Cesium.Color.PURPLE)
        //new Cesium.PolylineDashMaterialProperty({color: Cesium.Color.CYAN})

        clampToGround : true
    }
});
```

##### 1.7.1.11 Rectangle 
```javascript 
var rotation = Cesium.Math.toRadians(30);

function getRotationValue() {
    rotation += 0.005;
    return rotation;
}

var rectangle=viewer.entities.add({
    name:'rectangle',
    rectangle:{
        coordinates:Cesium.Rectangle.fromDegrees(-110.0, 20.0, -80.0, 25.0),
        material:Cesium.Color.RED.withAlpha(0.5),
        rotation: new Cesium.CallbackProperty(getRotationValue, false),
        stRotation: new Cesium.CallbackProperty(getRotationValue, false),
        extrudedHeight:300000.0,
        height:100000.0,
        outline:true,
        outlineColor:Cesium.Color.BLACK,
        classificationType : Cesium.ClassificationType.TERRAIN
    }
});
```

##### 1.7.12 Spheres and Ellipsoids 
```javascript
var blueEllipsoid = viewer.entities.add({
    name : 'Blue ellipsoid',
    position: Cesium.Cartesian3.fromDegrees(-114.0, 40.0, 300000.0),
    ellipsoid : {
        radii : new Cesium.Cartesian3(200000.0, 200000.0, 300000.0),
        fill:true,
        material : Cesium.Color.BLUE.withAlpha(0.5),
        outline : true,
        outlineColor : Cesium.Color.BLACK,
        slicePartitions : 24,
        stackPartitions : 36
    }
});
```

##### 1.7.1.13 Wall 
```javascript
var blueWall = viewer.entities.add({
    name : 'Blue wall with sawtooth heights and outline',
    wall : {
        positions : Cesium.Cartesian3.fromDegreesArray([-115.0, 50.0,
                                                        -112.5, 50.0,
                                                        -110.0, 50.0,
                                                        -107.5, 50.0,
                                                        -105.0, 50.0,
                                                        -102.5, 50.0,
                                                        -100.0, 50.0,
                                                        -97.5, 50.0,
                                                        -95.0, 50.0,
                                                        -92.5, 50.0,
                                                        -90.0, 50.0]),
        maximumHeights : [100000, 200000, 100000, 200000, 100000, 200000, 100000, 200000, 100000, 200000, 100000],
        minimumHeights : [0, 100000,  0, 100000, 0, 100000, 0, 100000, 0, 100000, 0],
        material : Cesium.Color.BLUE.withAlpha(0.5),
        outline : true,
        outlineColor : Cesium.Color.BLACK
    }
});
```

##### 1.7.1.14 zindex
在polygon描述中加入zindex，越大的排在越上面

##### 1.7.1.15 label\model\billboard
```javascript 
label:{
    text : 'Citizens Bank Park',
    font : '14pt monospace',
    style:Cesium.LabelStyle.FILL_AND_OUTLINE,
    outlineWidth:2,
    verticalOrigin:Cesium.VerticalOrigin.BOTTOM,
    pixelOffset:new Cesium.Cartesian2(0,-9)
}
model:{
    url:'xx'
}
billboard:{
    image:'xx',
    width:64,
    height:64
}
```
增加
```javascript
viewer.entities.add({
    id:'xxxid'
});
//-------or--------
var entity=viewer.entities.
```
查询
```javascript
var entity=viewer.entities.getById('uniqueId');
```
删除
```javascript
viewer.entities.remove(entity)
//----
viewer.entities.removeById('xxxid')
//-----
viewer.entities.removeAll()
```
实体集变化
```javascript
function onChanged(collection,added,removed,changed){
    var msg='Added ids';
    for(var i=0;i<added.length;++i)
        msg+='\n'+added[i].id;
    console.log(msg);
}
viewer.entities.collectionChanged.addEventListener(onChanged);
```

#### 1.7.2 eg 
**KML**

```javascript
var kmlOptions = {
    camera : viewer.scene.camera,
    canvas : viewer.scene.canvas,
    clampToGround : true
};
//http://catalog.opendata.city/dataset/pediacities-nyc-neighborhoods/resource/91778048-3c58-449c-a3f9-365ed203e914
var geocachePromise = Cesium.KmlDataSource.load('./Source/SampleData/sampleGeocacheLocations.kml', kmlOptions);
```

使用<code>KmlDataSource.load(options)</code>来读取点位数据<br> 
对于<code>KmlDataSource</code>，<code>camera</code>和<code>canvas</code>选项必须要配置，<code>clampToGround</code>控制数据是否贴地。贴地效果是常见的矢量数据可视化效果，保证数据紧贴地形起伏。

因为数据是异步加载的，所以这个函数实际返回一个<code>Promise</code>，最后使用<code>KmlDataSource</code>存储新创建的Entity。

<code>Promise</code>是一种异步处理机制，这里的“异步”是指需要在<code>.then</code>函数里操作数据，而不是直接在 <code>.load</code>函数之后立即操作。为了能在<code>scene</code>中使用这些载入的entity，只有当这个<code>promise</code>的<code>then回调中才可以把<code>KmlDataSource</code>添加到 <code>viewer.dataSources</code>。

```javascript
geocachePromise.then(function(dataSource) {
    // 把所有entities添加到viewer中显示
    viewer.dataSources.add(dataSource);
});
```

新加入到场景中的entity默认样式：<br>
1.单击它们时会在Infobox显示属性<br> 
2.双击它们的时候相机会转换为居中观察模式(lookAt)<br>
3.使用home按钮或infobox旁的相机按钮停止这种模式<br> 

也可以修改某些entity的显示样式

```javascript 
geocachePromise.then(function(dataSource) {
    // 把所有entities添加到viewer中显示
    viewer.dataSources.add(dataSource);
  
    // 获得entity列表
    var geocacheEntities = dataSource.entities.values;
  
    for (var i = 0; i < geocacheEntities.length; i++) {
        var entity = geocacheEntities[i];
        if (Cesium.defined(entity.billboard)) {
            // 对这个entity设置样式

            // 调整垂直方向的原点，保证图标里的针尖对着地表位置 
            entity.billboard.verticalOrigin = Cesium.VerticalOrigin.BOTTOM;
            // 去掉文字的显示
            entity.label = undefined;
            // 设置可见距离 控制只显示和相机一定距离内的点
            entity.billboard.distanceDisplayCondition = new Cesium.DistanceDisplayCondition(10.0, 20000.0);


            // 计算经度和纬度（角度表示）
            var cartographicPosition = Cesium.Cartographic.fromCartesian(entity.position.getValue(Cesium.JulianDate.now()));
            var longitude = Cesium.Math.toDegrees(cartographicPosition.longitude);
            var latitude = Cesium.Math.toDegrees(cartographicPosition.latitude);
            // 修改infobox描述信息  改成每个点的经纬度显示
            var description = '<table class="cesium-infoBox-defaultTable cesium-infoBox-defaultTable-lighter"><tbody>' +
                '<tr><th>' + "经度" + '</th><td>' + longitude.toFixed(5) + '</td></tr>' +
                '<tr><th>' + "纬度" + '</th><td>' + latitude.toFixed(5) + '</td></tr>' +
                '</tbody></table>';
            entity.description = description;
        }
    }
}); 
```

---

**GeoJson**

```javascript
var geojsonOptions = {
    clampToGround : true
};

// 从geojson文件加载行政区多边形边界数据
var neighborhoodsPromise = Cesium.GeoJsonDataSource.load('./Source/SampleData/neighborhoods.geojson', geojsonOptions);
 
var neighborhoods;

neighborhoodsPromise.then(function(dataSource) {
  
    viewer.dataSources.add(dataSource);

    neighborhoods = dataSource.entities;

    // 获取enity列表遍历
    var neighborhoodEntities = dataSource.entities.values;
    for (var i = 0; i < neighborhoodEntities.length; i++) {
        var entity = neighborhoodEntities[i];

        if (Cesium.defined(entity.polygon)) {
            // 设置样式代码

            // 把properties里的neighborhood设置到name
            entity.name = entity.properties.neighborhood;

            // 设置一个随机半透明颜色
            entity.polygon.material = Cesium.Color.fromRandom({
                red : 0.1,
                maximumGreen : 0.5,
                minimumBlue : 0.5,
                alpha : 0.6
            });

            // 设置这个属性让多边形贴地，ClassificationType.CESIUM_3D_TILE 是贴模型，ClassificationType.BOTH是贴模型和贴地
            entity.polygon.classificationType = Cesium.ClassificationType.TERRAIN;

            // 获取多边形的positions列表 并计算它的中心点
            var polyPositions = entity.polygon.hierarchy.getValue(Cesium.JulianDate.now()).positions;
            var polyCenter = Cesium.BoundingSphere.fromPoints(polyPositions).center;
            polyCenter = Cesium.Ellipsoid.WGS84.scaleToGeodeticSurface(polyCenter);
            entity.position = polyCenter;

            // 生成文字标注
            entity.label = {
                text : entity.name,
                showBackground : true,
                scale : 0.6,
                horizontalOrigin : Cesium.HorizontalOrigin.CENTER,
                verticalOrigin : Cesium.VerticalOrigin.BOTTOM,
                distanceDisplayCondition : new Cesium.DistanceDisplayCondition(10.0, 8000.0),
                disableDepthTestDistance : 100.0// 确保这个标注不会被其他对象盖住
            };
            
        }
    }
});
```

---

**czml**

```javascript
var dronePromise = Cesium.CzmlDsataSource.load('./Source/SampleData/SampleFlight.czml');



var drone;
dronePromise.then(function(dataSource) {
    viewer.dataSources.add(dataSource);

    // 使用id获取在CZML 数据中定义的无人机entity
    drone = dataSource.entities.getById('Aircraft/Aircraft1');

    // 附加一些三维模型
    drone.model = {
        uri : './Source/SampleData/Models/CesiumDrone.gltf',
        minimumPixelSize : 128,
        maximumScale : 1000,
        silhouetteColor : Cesium.Color.WHITE,
        silhouetteSize : 2
    };

    // 基于无人机轨迹的位置点，自动计算朝向
    drone.orientation = new Cesium.VelocityOrientationProperty(drone.position);

    // 光滑的路径插值
    drone.position.setInterpolationOptions({
        interpolationDegree : 3,
        interpolationAlgorithm : Cesium.HermitePolynomialApproximation
    });
});
```

---

### 1.8 3D Tile 
```javascript
var model=new Cesium.Cesium3DTileset({ url: Cesium.IonResource.fromAssetId(3839) });
var city = viewer.scene.primitives.add(model);

// 调整3dtile模型的高度，让他刚好放在地表
var heightOffset = -32;
city.readyPromise.then(function(tileset) {
    // Position tileset
    var boundingSphere = tileset.boundingSphere;
    //把数据当前的包围球转为Cartographic
    var cartographic = Cesium.Cartographic.fromCartesian(boundingSphere.center);
    var surface = Cesium.Cartesian3.fromRadians(cartographic.longitude, cartographic.latitude, 0.0);
    var offset = Cesium.Cartesian3.fromRadians(cartographic.longitude, cartographic.latitude, heightOffset);
    var translation = Cesium.Cartesian3.subtract(offset, surface, new Cesium.Cartesian3());
    tileset.modelMatrix = Cesium.Matrix4.fromTranslation(translation);
});

// var defaultStyle = new Cesium.Cesium3DTileStyle({
//     color : "color('white',0.3)",
//     show : true
// });

//设置让每个建筑style不同
var defaultStyle = new Cesium.Cesium3DTileStyle({
    color : {
        conditions : [
            ["${height} >= 300", "rgba(45, 0, 75, 0.5)"],
            ["${height} >= 200", "rgb(102, 71, 151)"],
            ["${height} >= 100", "rgb(170, 162, 204)"],
            ["${height} >= 50", "rgb(224, 226, 238)"],
            ["${height} >= 25", "rgb(252, 230, 200)"],
            ["${height} >= 10", "rgb(248, 176, 87)"],
            ["${height} >= 5", "rgb(198, 106, 11)"],
            ["true", "rgb(127, 59, 8)"]
        ]
    }
});

city.style = defaultStyle;

model.readyPromise.then(function(){
    viewer.zoomTo(model);
});
```

---

### 1.9 Interactivity
添加一些鼠标交互，当鼠标划过的时候，图标高亮。<br> 

有几种拾取：<br> 
- <code>scene.pick</code>:返回窗口坐标对应的图元的第一个对象。<br> 
- <code>scene.drillpick</code>:返回窗口坐标对应的所有对象列表。<br> 
- <code>globe.pick</code>:返回一条射线和地形的相交位置点。

要实现鼠标滑过的highlight效果，就要先创建一个鼠标事件处理器。<br> 
ScreenSpaceEventHandler是可以处理一系列的用户输入事件的处理器<br> 

设置一个回调函数来接受鼠标移动事件
```javascript
var handler=new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
handler.setInputAction(function(movement){},Cesium.ScreenSpaceEventType.MOUSE_MOVE);
```

下面设置highlight函数
```javascript
var handler=new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
var previousPickedEntity=undefined;
handler.setInputAction(function(movement){
    var pickedPrimitive=viewer.scene.pick(movement.endPosition);
    var pickedEntity=(Cesium.defined(pickedPrimitive))?pickedPrimitive.id:undefined;

    //取消上一个高光对象的高光效果
    if(Cesium.defined(previousPickedEntity)){
        previousPickedEntity.billboard.scale=1.0;
        previousPickedEntity.billboard.color=Cesium.Color.WHITE;
    }

    //拾取到一个billboard对象，则目前鼠标在一个图标上了
    if(Cesium.defined(pickedEntity)&&Cesium.defined(pickedEntity.billboard)){
        pickedEntity.billboard.scale=2.0;
        pickedEntity.billboard.color=Cesium.Color.ORANGERED;
    }

},Cesium.ScreenSpaceEventType.MOUSE_MOVE);
```

---

### 1.10 Camera Modes 
相机模式
```javascript
var freeModeElement = document.getElementById('freeMode');
var droneModeElement = document.getElementById('droneMode');

function setViewMode(){
    if(droneModeElement.checked){
        viewer.trackedEntity=drone;
    }else{
        viewer.trackedEntity=undefined;
        viewer.scene.camera.flyTo(homeCameraView);
    }
}

freeModeElement.addEventListener('change', setCameraMode);
droneModeElement.addEventListener('change', setCameraMode);

viewer.trackedEntityChanged.addEventListener(function() {
    if (viewer.trackedEntity === drone) {
        freeModeElement.checked = false;
        droneModeElement.checked = true;
    }
});
```

---

### 1.11 Extras
因为3D Tiles数据可能不是瞬间载入，可以添加一个载入指示器，当所有切片都载入后隐藏。

```javascript
// 当城市数据初始化完成后，移除加载指示器
var loadingIndicator = document.getElementById('loadingIndicator');
loadingIndicator.style.display = 'block';
city.readyPromise.then(function () {
    loadingIndicator.style.display = 'none';
});
```