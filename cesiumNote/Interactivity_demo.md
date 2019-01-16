# Interactivity_demo 
### picking in action 
#### picking demo 
PS:html上需要加上bucket.css 及 id为loadingOverlay与toolbar的div

```javascript 
var viewer = new Cesium.Viewer('cesiumContainer', {
    selectionIndicator : false,
    infoBox : false
});

var scene = viewer.scene;
if (!scene.pickPositionSupported) {
    console.log('This browser does not support pickPosition.');
}

var handler;

Sandcastle.addDefaultToolbarButton('Show Cartographic Position on Mouse Over', function() {
    var entity = viewer.entities.add({
        label : {
            show : false,
            showBackground : false,
            font : '14px monospace',
            horizontalOrigin : Cesium.HorizontalOrigin.RIGHT,
            verticalOrigin : Cesium.VerticalOrigin.BOTTOM,
            pixelOffset : new Cesium.Cartesian2(50, 50)
        }
    });

    // Mouse over the globe to see the cartographic position
    handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
    handler.setInputAction(function(movement) {
        
        //获取椭球面坐标
        var cartesian = viewer.camera.pickEllipsoid(movement.endPosition, scene.globe.ellipsoid);
        
        if (cartesian && entity.label.show) {
            var cartographic = Cesium.Cartographic.fromCartesian(cartesian);
            var longitudeString = Cesium.Math.toDegrees(cartographic.longitude).toFixed(2);
            var latitudeString = Cesium.Math.toDegrees(cartographic.latitude).toFixed(2);

            entity.position = cartesian;
            entity.label.show = true;
            entity.label.text =
                'Lon: ' + ('   ' + longitudeString).slice(-7) + '\u00B0' +
                '\nLat: ' + ('   ' + latitudeString).slice(-7) + '\u00B0';
        } else {
            entity.label.show = false;
        }
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
});

Sandcastle.addToolbarButton('Pick Entity', function() {
    var entity = viewer.entities.add({
        position : Cesium.Cartesian3.fromDegrees(-75.59777, 40.03883),
        billboard : {
            image : '../images/Cesium_Logo_overlay.png'
        }
    });

    // If the mouse is over the billboard, change its scale and color
    handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
    handler.setInputAction(function(movement) {
        var pickedObject = scene.pick(movement.endPosition);
        if (Cesium.defined(pickedObject) && (pickedObject.id === entity)) {
            entity.billboard.scale = 2.0;
            entity.billboard.color = Cesium.Color.YELLOW;
        } else {
            entity.billboard.scale = 1.0;
            entity.billboard.color = Cesium.Color.WHITE;
        }
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
});

Sandcastle.addToolbarButton('Drill-Down Picking', function() {
    var pickedEntities = new Cesium.EntityCollection();
    var pickColor = Cesium.Color.YELLOW.withAlpha(0.5);
    function makeProperty(entity, color) {
        var colorProperty = new Cesium.CallbackProperty(function(time, result) {
            if (pickedEntities.contains(entity)) {
                return pickColor.clone(result);
            }
            return color.clone(result);
        }, false);

        entity.polygon.material = new Cesium.ColorMaterialProperty(colorProperty);
    }

    var red = viewer.entities.add({
        polygon : {
            hierarchy : Cesium.Cartesian3.fromDegreesArray([-70.0, 30.0,
                                                            -60.0, 30.0,
                                                            -60.0, 40.0,
                                                            -70.0, 40.0]),
            height : 0
        }
    });
    makeProperty(red, Cesium.Color.RED.withAlpha(0.5));

    var blue = viewer.entities.add({
        polygon : {
            hierarchy : Cesium.Cartesian3.fromDegreesArray([-75.0, 34.0,
                                                            -63.0, 34.0,
                                                            -63.0, 40.0,
                                                            -75.0, 40.0]),
            height : 0
        }
    });
    makeProperty(blue, Cesium.Color.BLUE.withAlpha(0.5));

    var green = viewer.entities.add({
        polygon : {
            hierarchy : Cesium.Cartesian3.fromDegreesArray([-67.0, 36.0,
                                                            -55.0, 36.0,
                                                            -55.0, 30.0,
                                                            -67.0, 30.0]),
            height : 0
        }
    });
    makeProperty(green, Cesium.Color.GREEN.withAlpha(0.5));

    // Move the primitive that the mouse is over to the top.
    handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
    handler.setInputAction(function(movement) {
        // get an array of all primitives at the mouse position
        var pickedObjects = scene.drillPick(movement.endPosition);
        if (Cesium.defined(pickedObjects)) {
            //Update the collection of picked entities.
            pickedEntities.removeAll();
            for (var i = 0; i < pickedObjects.length; ++i) {
                var entity = pickedObjects[i].id;
                pickedEntities.add(entity);
            }
        }

    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
});

Sandcastle.addToolbarButton('Pick position', function() {
    var modelEntity = viewer.entities.add({
        name : 'milktruck',
        position : Cesium.Cartesian3.fromDegrees(-123.0744619, 44.0503706),
        model : {
            uri : '../../../../Apps/SampleData/models/CesiumMilkTruck/CesiumMilkTruck-kmc.gltf'
        }
    });
    viewer.zoomTo(modelEntity);

    var labelEntity = viewer.entities.add({
        label : {
            show : false,
            showBackground : true,
            font : '14px monospace',
            horizontalOrigin : Cesium.HorizontalOrigin.LEFT,
            verticalOrigin : Cesium.VerticalOrigin.TOP,
            pixelOffset : new Cesium.Cartesian2(15, 0)
        }
    });

    // Mouse over the globe to see the cartographic position
    handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
    handler.setInputAction(function(movement) {

        var foundPosition = false;

        var scene = viewer.scene;
        if (scene.mode !== Cesium.SceneMode.MORPHING) {
            var pickedObject = scene.pick(movement.endPosition);
            if (scene.pickPositionSupported && Cesium.defined(pickedObject) && pickedObject.id === modelEntity) {
                var cartesian = viewer.scene.pickPosition(movement.endPosition);

                if (Cesium.defined(cartesian)) {
                    var cartographic = Cesium.Cartographic.fromCartesian(cartesian);
                    var longitudeString = Cesium.Math.toDegrees(cartographic.longitude).toFixed(2);
                    var latitudeString = Cesium.Math.toDegrees(cartographic.latitude).toFixed(2);
                    var heightString = cartographic.height.toFixed(2);

                    labelEntity.position = cartesian;
                    labelEntity.label.show = true;
                    labelEntity.label.text =
                        'Lon: ' + ('   ' + longitudeString).slice(-7) + '\u00B0' +
                        '\nLat: ' + ('   ' + latitudeString).slice(-7) + '\u00B0' +
                        '\nAlt: ' + ('   ' + heightString).slice(-7) + 'm';

                    labelEntity.label.eyeOffset = new Cesium.Cartesian3(0.0, 0.0, -cartographic.height * (scene.mode === Cesium.SceneMode.SCENE2D ? 1.5 : 1.0));

                    foundPosition = true;
                }
            }
        }

        if (!foundPosition) {
            labelEntity.label.show = false;
        }
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
});

Sandcastle.reset = function() {
    viewer.entities.removeAll();
    handler = handler && handler.destroy();
};

```

#### 3D Tiles Feature Picking Demo
```javascript
var viewer=new Cesium.Viewer('cesiumContainer',{
    terrainProvider:Cesium.createWorldTerrain()
})

viewer.scene.globe.depthTestAgainstTerrain=true;

//设置初始照相机视角
var initialPosition=Cesium.Cartesian3.fromDegrees(-74.01881302800248, 40.69114333714821, 753);
var initialOrientation=new Cesium.HeadingPitchRange.fromDegrees(21.27879878293835, -21.34390550872461, 0.0716951918898415);
viewer.scene.camera.setView({
    destination:initialPosition,
    orientation:initialOrientation,
    endTransform:Cesium.Matrix4.IDENTITY
})

var tileset=new Cesium.Cesium3DTileset({url:Cesium.IonResource.fromAssetId(5741)});
viewer.scene.primitives.add(tileset);

//html overlay 来显示特征名字
var nameOverlay=document.createElement('div');
viewer.container.appendChild(nameOverlay);
nameOverlay.className='backdrop';
nameOverlay.style.display='none';
nameOverlay.style.position='absolute';
nameOverlay.style.bottom='0';
nameOverlay.style.left='0'
nameOverlay.style['pointer-events']='none';
nameOverlay.style.padding='4px';
nameOverlay.style.backgroundColor='black';

//现在点击的特征信息
var selected={
    feature:undefined,
    originColor:new Cesium.Color()
};

//一个实体对象，它将保存有关信息框显示当前所选功能的信息。
var selectedEntity=new Cesium.Entity();

//获取默认的左键单击处理程序，用于在左键单击时未选中某个功能
var clickHandler=viewer.screenSpaceEventHandler.getInputAction(Cesium.ScreenSpaceEventType);

//如果支持轮廓，则在鼠标悬停时使用蓝色轮廓，在鼠标单击时使用绿色轮廓。
//如果不支持轮廓，请在鼠标悬停时将fill颜色更改为黄色，单击时更改为绿色。
if(Cesium.PostProcessStageLibrary.isSilhoutteSupported(viewer.scene)){
    //supported
    var silhouetteBlue=Cesium.PostProcessStageLibrary.createEdgeDetectionStage();
    silhouetteBlue.uniforms.color=Cesium.Color.BLUE;
    silhouetteBlue.uniforms.length=0.01;
    silhouetteBlue.selected=[];

    var silhouetteGreen=Cesium.PostProcessStageLibrary.createEdgeDetectionStage();
    silhouetteGreen.uniforms.color=Cesium.Color.LIME;
    silhouetteGreen.uniforms.length=0.01;
    silhouetteGreen.selected=[];

    viewer.scene.PostProcessStageLibrary.add(Cesium.PostProcessStageLibrary.createSilhouetteStage([silhouetteBlue,silhouetteGreen]));

    //画轮廓
    viewer.screenSpaceEventHandler.setInputAction(function onMouseMove(movement){
        //如果有一个特征之前已经高亮了，取消它的高亮
        silhouetteBlue.selected=[]

        //找一个新特征 
        var pickedFeature=viewer.scene.pick(movement.endPosition);
        if(!Cesium.defined(pickedFeature)){
            nameOverlay.style.display='none';
            return;
        }

        //overlay content
        nameOverlay.style.display='block';
        nameOverlay.style.bottom=viewer.canvas.clientHeight-movement.endPosition.y+'px';
        nameOverlay.style.left=movement.endPosition.x+'px';

        var name=pickedFeature.getProperty('name');
        if(!Cesium.defined(name)){
            name=pickedFeature.getProperty('id');
        }
        nameOverlay.textContent=name;

        //如果它还没被选择的话高亮这个特征
        if(pickedFeature!=selected.feature){
            silhouetteBlue.selected=[pickedFeature];
        }
    },Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    //轮廓出一个被选择的特征，并在信息框中显示元数据。
    viewer.screenSpaceEventHandler.setInputAction(function onLeftClick(movement){
        silhouetteGreen.selected=[];

        var pickedFeature=viewer.scene.pick(movement.position);
        if(!Cesium.defined(pickedFeature)){
            clickHandler(movement);
            return;
        }

        if(silhouetteGreen.selected[0]==pickedFeature){
            return;
        }

        //保存所选特征的原有颜色
        var highlightedFeature=silhouetteBlue.selected[0];
        if(pickedFeature==highlightedFeature){
            silhouetteBlue.selected=[];
        }

        //高亮新选择的特征
        silhouetteGreen.selected=[pickedFeature];

        //设置功能信息框说明
        var featureName = pickedFeature.getProperty('name');
        selectedEntity.name = featureName;
        selectedEntity.description = 'Loading <div class="cesium-infoBox-loading"></div>';
        viewer.selectedEntity = selectedEntity;
        selectedEntity.description = '<table class="cesium-infoBox-defaultTable"><tbody>' +
                                     '<tr><th>BIN</th><td>' + pickedFeature.getProperty('BIN') + '</td></tr>' +
                                     '<tr><th>DOITT ID</th><td>' + pickedFeature.getProperty('DOITT_ID') + '</td></tr>' +
                                     '<tr><th>SOURCE ID</th><td>' + pickedFeature.getProperty('SOURCE_ID') + '</td></tr>' +
                                     '</tbody></table>';
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

}else{
    // Silhouettes are not supported. Instead, change the feature color.

    // Information about the currently highlighted feature
    var highlighted = {
        feature : undefined,
        originalColor : new Cesium.Color()
    };

    // Color a feature yellow on hover.
    viewer.screenSpaceEventHandler.setInputAction(function onMouseMove(movement) {
        // If a feature was previously highlighted, undo the highlight
        if (Cesium.defined(highlighted.feature)) {
            highlighted.feature.color = highlighted.originalColor;
            highlighted.feature = undefined;
        }
        // Pick a new feature
        var pickedFeature = viewer.scene.pick(movement.endPosition);
        if (!Cesium.defined(pickedFeature)) {
            nameOverlay.style.display = 'none';
            return;
        }
        // A feature was picked, so show it's overlay content
        nameOverlay.style.display = 'block';
        nameOverlay.style.bottom = viewer.canvas.clientHeight - movement.endPosition.y + 'px';
        nameOverlay.style.left = movement.endPosition.x + 'px';
        var name = pickedFeature.getProperty('name');
        if (!Cesium.defined(name)) {
            name = pickedFeature.getProperty('id');
        }
        nameOverlay.textContent = name;
        // Highlight the feature if it's not already selected.
        if (pickedFeature !== selected.feature) {
            highlighted.feature = pickedFeature;
            Cesium.Color.clone(pickedFeature.color, highlighted.originalColor);
            pickedFeature.color = Cesium.Color.YELLOW;
        }
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

    // Color a feature on selection and show metadata in the InfoBox.
    viewer.screenSpaceEventHandler.setInputAction(function onLeftClick(movement) {
        // If a feature was previously selected, undo the highlight
        if (Cesium.defined(selected.feature)) {
            selected.feature.color = selected.originalColor;
            selected.feature = undefined;
        }
        // Pick a new feature
        var pickedFeature = viewer.scene.pick(movement.position);
        if (!Cesium.defined(pickedFeature)) {
            clickHandler(movement);
            return;
        }
        // Select the feature if it's not already selected
        if (selected.feature === pickedFeature) {
            return;
        }
        selected.feature = pickedFeature;
        // Save the selected feature's original color
        if (pickedFeature === highlighted.feature) {
            Cesium.Color.clone(highlighted.originalColor, selected.originalColor);
            highlighted.feature = undefined;
        } else {
            Cesium.Color.clone(pickedFeature.color, selected.originalColor);
        }
        // Highlight newly selected feature
        pickedFeature.color = Cesium.Color.LIME;
        // Set feature infobox description
        var featureName = pickedFeature.getProperty('name');
        selectedEntity.name = featureName;
        selectedEntity.description = 'Loading <div class="cesium-infoBox-loading"></div>';
        viewer.selectedEntity = selectedEntity;
        selectedEntity.description = '<table class="cesium-infoBox-defaultTable"><tbody>' +
                                     '<tr><th>BIN</th><td>' + pickedFeature.getProperty('BIN') + '</td></tr>' +
                                     '<tr><th>DOITT ID</th><td>' + pickedFeature.getProperty('DOITT_ID') + '</td></tr>' +
                                     '<tr><th>SOURCE ID</th><td>' + pickedFeature.getProperty('SOURCE_ID') + '</td></tr>' +
                                     '<tr><th>Longitude</th><td>' + pickedFeature.getProperty('longitude') + '</td></tr>' +
                                     '<tr><th>Latitude</th><td>' + pickedFeature.getProperty('latitude') + '</td></tr>' +
                                     '<tr><th>Height</th><td>' + pickedFeature.getProperty('height') + '</td></tr>' +
                                     '<tr><th>Terrain Height (Ellipsoid)</th><td>' + pickedFeature.getProperty('TerrainHeight') + '</td></tr>' +
                                     '</tbody></table>';
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
}
````