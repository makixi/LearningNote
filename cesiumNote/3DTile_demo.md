# 3D Tile DEMO 

### 3D Tiles Adjust Height 
模型平移
```javascript
function changeHeight(height){
    height=Number(height);
    if(isNaN(height)){
        return;
    }

    var cartographic=Cesium.Cartographic.fromCartesian(tileset.boundingSphere.center);
    var surface=Cesium.Cartesian3.fromRadians(cartographic.longitude,cartographic.latitude,cartographic.height);
    var offset=Cesium.Cartesian3.fromRadians(cartographic.longitude,cartographic.latitude,height);
    var translation=Cesium.Cartesian3.subtract(offset,surface,new Cesium.Cartesian3());
    tileset.modelMatrix=Cesium.Matrix4.fromTranslation(translation);
}
```

模型旋转

```javascript
var m=tileset.modelMatrix;
//rotate为旋转角度
var m1=Cesium.Matrix3.fromRotationX(Cesium.Math.toRadians(RotateX));
Cesium.Matrix4.multiplyByMatrix3(m,m1,m);
tileset.modelMatrix=m;
```