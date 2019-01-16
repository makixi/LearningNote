//can't render anything before the level zero tiles exist
if(!defined(primitive._levelZeroTiles)){
    if(primitive._tileProvider.ready){
        var tilingScheme=primitive._tileProvider.tilingScheme;
        primitive._levelZeroTiles=QuadtreeTile.createLevelZeroTiles(tilingScheme);
    }else{
        //nothing to do until the provider is ready
        return;
    }
}

//judge the tile could be seen at this condition
if(tile.renderable && tileProvider.computeTileVisibility(tile,frameState,occluders)!==Visibility.NONE){
    traversalQueue.enqueue(tile);
}


//Traversing quadtree
while(defined((tile=traversalQueue.dequeue()))){
    if(screenSpaceError(primitive,frameState,tile)<primitive.maximumScreenSpaceError){
        addTileToRenderList(primitive,tile);
    }else if(queueChildrenLoadAndDetermineIfChildrenAreAllRenderable(primitive,tile)){
        var children=tile.children;
        for(i=0,len=children.length;i<len;++i){
            if(tileProvider.computeTileVisibility(children[i],frameState,occluders)!=Visibility.NONE){
                traversalQueue.enqueue(children[i]);
            }else{
                ++debugg.tilesCulled;
            }
        }
    }else{
        addTileToRenderList(primitive,tile);
    }
}