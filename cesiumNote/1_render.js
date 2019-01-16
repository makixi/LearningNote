//------begin to loop

function startRenderLoop(widget){
    widget._renderLoopRunning=true;

    var lastFrameTime=0;
    function render(frameTime);

    requestAnimationFrame(render);//animation
}


//requestAnimationFrame(render)最终调用的方法
Scene.prototype.render=function(time){
    try{
        render(this,time);
    }catch(error){
        this._renderError.raiseEvent(this,error);

        if(this.rethrowRenderErrors){
            throw error; 
        }
    }
}


//function render(scen,time) is a little long
//terrain and imagery are finished between beginFrame and endFrame 

if(defined(scene,globe)){
    scene.globe.beginFrame(frameState);
}
updateEnvironment(scene);
//responsibel for data scheduling(management) 
updateAndExecuteCommands(scene,passState,defaultValue(scene,backgroundColor,Color.BLACK));
resolveFramebuffers(scene,passState);
executeOverlayCommands(scene,passState);
if(defined(scene,globe)){
    scene.globe.endFrame(frameState);
}

