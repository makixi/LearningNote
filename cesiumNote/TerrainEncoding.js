var aaBox=new AxisAlignedBoundingBox(minimum,maximum,center);
var encoding=new TerrainEncoding(aaBox,hMin,maximumHeight,fromENU,hasVertexNormals);

AttributeCompression.compressTextureCoordinates=function(textureCoordinates){
    var x=textureCoordinates.x==1.0?4095.0:(textureCoordinates.x*4096.0)|0;
    var y=textureCoordinates.y==1.0?4095.0:(textureCoordinates.y*4096.0)|0;
    return 4096*x+y;
};