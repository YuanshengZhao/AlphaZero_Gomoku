var a0eng=new A0ENGINE();
function roundToTwo(num) {    
return num.toFixed(3);
}
async function LoadEngine()
{
    a0eng.model=await tf.loadGraphModel("https://raw.githubusercontent.com/YuanshengZhao/AlphaZero_Gomoku/weights/RNGjs_web/model.json");
    console.log("tf backend:",tf.getBackend());
    document.getElementById("engineinfo").innerHTML="cpuct = 1.25; c_base = 19652; FPU_reduction = "+roundToTwo(fpuReduction-1)+" | "+roundToTwo(fpuReductionRoot-1)+"; hash_size = "+a0eng.zobrist.hash_size+"; tf_backend = "+tf.getBackend();
}
LoadEngine();
var checkautoplay = document.getElementById("autoplay");
var selectN_node = document.getElementById("N_node");
var selectTemperature = document.getElementById("Temperature");
var temperatureG=0.5;
max_cache=4;

const infop=document.getElementById("info");
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");
canvas.style.width = clientSz+"px";
canvas.style.height = boardHt+"px";
canvas.width = Math.ceil(clientSz*sdpi);
canvas.height = Math.ceil(boardHt*sdpi);
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.scale(sdpi,sdpi);
ctx.font =hdx+"px \'Ubuntu\', sans-serif";
// var eval_values=new Array(225);
var eval_values=new Array(maxPossibleMoves);
var move_history=[];
for(var i=0;i<maxPossibleMoves;i++) 
{
    move_history[i]=[0,0];
}

const lw1=lw/4
console.log("lw",lw,lw1)
function redraw_canvas()
{
    // console.log("repaint");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.lineWidth = lw1;
    var brwd=Math.min(hdx,~~(dx*16/(move_count+1)))
    var ddy
    for(var ii=0;ii<move_count;++ii)
    {
        ddy=ii%2==0? ((eval_values[ii]-0.5)*2*barlength):((0.5-eval_values[ii])*2*barlength);
        ctx.fillStyle=ddy>0? "#595959": "#b2b2b2";
        ctx.beginPath();
        ctx.rect(dx+brwd*ii,barcenter,brwd,-ddy);
        ctx.fill();
        ctx.stroke();
    }
    ctx.fillStyle = "#000000";
    ctx.beginPath();
    for(var i=0;i<15;++i)
    {
        ctx.moveTo(dx*2,    dx*(i+2))
        ctx.lineTo(dx*16,   dx*(i+2))
        ctx.moveTo(dx*(i+2),dx*2)
        ctx.lineTo(dx*(i+2),dx*16)
        ctx.fillText((15-i).toString(),dx,dx*(i+2))
        ctx.fillText((15-i).toString(),dx*17,dx*(i+2))
        ctx.fillText(String.fromCharCode(65+i).toString(),dx*(i+2),dx)
        ctx.fillText(String.fromCharCode(65+i).toString(),dx*(i+2),dx*17)
    }
    ctx.moveTo(dx,barcenter-barlength);ctx.lineTo(dx*17,barcenter-barlength);
    ctx.moveTo(dx,barcenter+barlength);ctx.lineTo(dx*17,barcenter+barlength);
    ctx.fillStyle = "#000000";
    ctx.stroke();
    ctx.fillRect(dx*(7 +2)-sar,dx*(7 +2)-sar,2*sar,2*sar)
    ctx.fillRect(dx*(3 +2)-sar,dx*(3 +2)-sar,2*sar,2*sar)
    ctx.fillRect(dx*(3 +2)-sar,dx*(11+2)-sar,2*sar,2*sar)
    ctx.fillRect(dx*(11+2)-sar,dx*(3 +2)-sar,2*sar,2*sar)
    ctx.fillRect(dx*(11+2)-sar,dx*(11+2)-sar,2*sar,2*sar)
    ctx.lineWidth = lw;
    for(var i=0;i<move_count;i+=2)
    {
        ctx.beginPath();
        ctx.arc(dx*(move_history[i][0]+2),dx*(move_history[i][1]+2),ds,0,6.29)
        ctx.stroke()
        ctx.fill();
    }
    ctx.fillStyle = "#ffffff";
    for(var i=0;i<move_count;i+=2)
    {
        ctx.fillText(i+1,dx*(move_history[i][0]+2),dx*(move_history[i][1]+2))
    }
    for(var i=1;i<move_count;i+=2)
    {
        ctx.beginPath();
        ctx.arc(dx*(move_history[i][0]+2),dx*(move_history[i][1]+2),ds,0,6.29)
        ctx.stroke()
        ctx.fill();
    }
    ctx.fillStyle = "#000000";
    for(var i=1;i<move_count;i+=2)
    {
        ctx.fillText(i+1,dx*(move_history[i][0]+2),dx*(move_history[i][1]+2))
    }
    ctx.fillStyle = "#FF4000";
    if(move_count>0) ctx.fillText(move_count,dx*(move_history[move_count-1][0]+2),dx*(move_history[move_count-1][1]+2))
}
redraw_canvas();
canvas.addEventListener("click",PlayStone);
var gameover=false;
function PlayStone(event)
{
    if(playStoneXY(~~((event.offsetY+hdx)/dx)-2,~~((event.offsetX+hdx)/dx)-2)>=0 && !gameover && checkautoplay.checked)
    {
        canvas.removeEventListener("click",PlayStone);
        DelayedCP();
    }
}
function playStoneXY(px,py,evalu=0.5)
{
    if(px<0 || px>14 || py<0 || py>14)
        return -1;
    if(board[px][py]!=0)
    {
        console.log("Non empty!",board[px][py]);
        return -1;
    }
    else
    {
        console.log("play",px,py)
        move_history[move_count]=[py,px];
        eval_values[move_count]=evalu;
        applyMoveXY(px,py,a0eng);
        redraw_canvas();
        gameover=(winLossDraw()>-.5)
        if(gameover)
        {
            ctx.globalAlpha = 0.5;
            ctx.font =~~(hdx*4)+"px \'Ubuntu\', sans-serif";
            ctx.fillStyle = "#FF00FF";
            ctx.fillText("Game over!",dx*9,dx*9);
            ctx.font =hdx+"px \'Ubuntu\', sans-serif";
            ctx.globalAlpha = 1;
            canvas.removeEventListener("click",PlayStone);
            return 1;
        }
    }
    return 0
}
function newGame()
{
    clearAll(a0eng);
    redraw_canvas();
    if(gameover)
    {
        canvas.addEventListener("click",PlayStone);
        gameover=false;
    }
    infop.innerHTML="New game";
    console.log("new game")
}
function takeBackG()
{
    if(move_count==0)
    {
        infop.innerHTML="Cannot take back"
        return;
    }
    takeBack(a0eng);
    redraw_canvas();
    infop.innerHTML="Take back";
    console.log("take back");
    if(gameover)
    {
        canvas.addEventListener("click",PlayStone);
        gameover=false;
    }
}
function DelayedCP()
{
    if(gameover) 
    {
        infop.innerHTML="Cannot call comp_play"
        return;
    }
    infop.innerHTML="Evaluating..."
    setTimeout(function(){ComputerPlay();},100);
}
function ComputerPlay()
{
    var t0_ = performance.now();
    var eg_act=run_mcts(a0eng,temperatureG);
    var t1_ = performance.now();
    infop.innerHTML="evaluation = "+roundToTwo(engine_out_value)+"; seldepth = "+(maxdepth-1)+"; time = "+parseInt(t1_ - t0_)+" ms";
    console.log("comp play v "+engine_out_value+" "+(t1_-t0_)+"|"+engine_t+" ms.");
    playStoneXY(~~(eg_act/15),eg_act%15,engine_out_value);
    if(! gameover) canvas.addEventListener("click",PlayStone);
}
function setNNodeG()
{
    var nnd=selectN_node.value;
    if(nnd<=100) max_cache=4;
    else max_cache=16;
    setNum_sml(parseInt(selectN_node.value));
}
function setTG()
{
    temperatureG=parseFloat(selectTemperature.value);
    console.log(temperatureG);
}