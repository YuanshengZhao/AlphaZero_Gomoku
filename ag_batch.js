// import * as tf from '@tensorflow/tfjs-node';
// import Random from "random-js"

var board=[];
for(var i=0;i<15;i++) 
{
    board[i]=[];
    for(var j=0;j<15;j++) 
    {
        board[i][j]=0;
    }
}

const engine = Random.MersenneTwister19937.autoSeed()
const maxPossibleMoves=225;
var side2move=0;
var movelist=new Array(maxPossibleMoves);
var move_count=0;
var num_simul=10;
var max_cache=16;

function printBoard(mboard=board)
{
    var txt="";
    txt+="   ";
    for(var ii=0;ii<15;ii++)
        txt=txt+" "+(ii%10).toString()+" ";
    txt=txt+"\n";
    for(var ii=0;ii<15;ii++)
    {
        txt=txt+" "+(ii%10).toString()+" ";
        for(var jj=0;jj<15;jj++)
        {
            if(mboard[ii][jj]==0)
            {
                if((ii==7 && jj==7)||((ii==3||ii==11) && (jj==3||jj==11)))
                    txt=txt+" + ";
                else
                    txt=txt+" - ";
            }
            else if(mboard[ii][jj]==1)
                txt=txt+" O ";
            else if(mboard[ii][jj]==-1)
                txt=txt+" X ";
            else
                txt=txt+" E ";
        }
            txt=txt+" "+(ii%10).toString()+"\n";
    }
    txt=txt+"   ";
    for(var ii=0;ii<15;ii++)
        txt=txt+" "+(ii%10).toString()+" ";
    txt=txt+"\n";
    console.log(txt);
}
function printBoard_bu(mboard)
{
    var txt="";
    txt+="   ";
    for(var ii=0;ii<15;ii++)
        txt=txt+" "+(ii%10).toString()+" ";
    txt=txt+"\n";
    for(var ii=0;ii<15;ii++)
    {
        txt=txt+" "+(ii%10).toString()+" ";
        for(var jj=0;jj<15;jj++)
        {
            if(mboard[ii][jj][0]==0 && mboard[ii][jj][1]==0)
            {
                if((ii==7 && jj==7)||((ii==3||ii==11) && (jj==3||jj==11)))
                    txt=txt+" + ";
                else
                    txt=txt+" - ";
            }
            else if(mboard[ii][jj][0]==1 && mboard[ii][jj][1]==0)
                txt=txt+" O ";
            else if(mboard[ii][jj][1]==1 && mboard[ii][jj][0]==0)
                txt=txt+" X ";
            else
                txt=txt+" E ";
        }
            txt=txt+" "+(ii%10).toString()+"\n";
    }
    txt=txt+"   ";
    for(var ii=0;ii<15;ii++)
        txt=txt+" "+(ii%10).toString()+" ";
    txt=txt+"\n";
    console.log(txt);
}

class NODE
{
    constructor()
    {
        this.visit_count=0;
        this.to_play=null;
        this.prior=null;
        this.value_sum=0.0;
        this.actions=new Array(maxPossibleMoves);
        this.children=new Array(maxPossibleMoves);
        this.num_child=0;
    }

    set_state(sidem)
    {
        this.visit_count=0;
        this.to_play=sidem;
        this.value_sum=0.0;
        this.num_child=0;
    }
    expanded()
    {
        return (this.num_child>0);
    }
    value()
    {
        //unvisited node call result in error!!!
        return this.value_sum/this.visit_count;
    }
};


var node_capacity=(num_simul+1)*224>500? (num_simul+1)*224 : 500;
var nodes=[];

for(var i=0;i<node_capacity;++i)
    nodes.push(new NODE());

function setNum_sml(nsml)
{
    var sz;
    num_simul=nsml;
    if((sz=(num_simul+1)*224)>node_capacity)
    {
        for(var i=node_capacity;i<sz;++i)
            nodes.push(new NODE());
        node_capacity=sz;
    }
    console.log("num_simul = %d\n",num_simul);
}

var rootnode= new NODE();
var nodes_used=0;

function winLossDraw()//return O wins or X wins
{
    var sd=2*side2move-1;
    var posx=~~(movelist[move_count-1]/15), posy=movelist[move_count-1]%15;

    var lx=1,lg=1;
    while(posx-lx>=0 && board[posx-lx][posy]==sd){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 && board[posx+lx][posy]==sd){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posy-lx>=0 && board[posx][posy-lx]==sd){lg+=1; lx+=1;}
    lx=1;
    while(posy+lx<15 && board[posx][posy+lx]==sd){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posx-lx>=0 && posy-lx>=0 && board[posx-lx][posy-lx]==sd){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 && posy+lx<15 && board[posx+lx][posy+lx]==sd){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posx-lx>=0 && posy+lx<15 && board[posx-lx][posy+lx]==sd){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 && posy-lx>=0 && board[posx+lx][posy-lx]==sd){lg+=1; lx+=1;}
    if(lg>=5) return side2move;
    
    if(move_count==maxPossibleMoves) return 0.5;
    return -1;
}

class ZOBRIST
{
    constructor(sze)
    {
        var lg=1;
        while(sze>>lg) lg++;
        this.hash_size=1<<(lg-1);
        console.log("Hash size = "+(this.hash_size).toString()+"\n");
        var szz=2*maxPossibleMoves
        this.table=new Array(szz);
        for(var i=0;i<szz;i++)
            this.table[i]=Random.integer(0,this.hash_size-1)(engine);
        this.initkey=Random.integer(0,this.hash_size-1)(engine);
        this.currentkey=this.initkey;
        this.values=new Array((maxPossibleMoves+1)*this.hash_size);
        szz=maxPossibleMoves*this.hash_size
        this.boardcheck=new Array(szz);
        for(var i=0;i<szz;i++)
            this.boardcheck[i]=-1;
    }

    updatekey(pos)
    {
        this.currentkey^=this.table[pos];
    }
    clearkey()
    {
        this.currentkey=this.initkey;
    }
    
    getValue(boardX)
    {
        var ixx=this.currentkey*maxPossibleMoves;
        for(var i=0;i<15;i++) 
        {
            for(var j=0;j<15;j++) 
            {
                if(boardX[i][j]!=this.boardcheck[ixx++]) return null;
            }
        }
        return this.values.slice(this.currentkey*(maxPossibleMoves+1),(this.currentkey+1)*(maxPossibleMoves+1));
    }
    
    setValue(key,rst_in, board_in,sd)
    {
        var ixx=key*(maxPossibleMoves+1);
        for(var i=0;i<=maxPossibleMoves;++i)
        this.values[ixx++]=rst_in[i];
        ixx=key*(maxPossibleMoves);
        var rsd=1-sd;
        for(var i=0;i<15;i++) 
            {
                for(var j=0;j<15;j++) 
                {
                    this.boardcheck[ixx++]=board_in[i][j][sd]-board_in[i][j][rsd];
                }
            }
    }
};

var engine_t=0;

class A0ENGINE
{
    constructor()
    {
        this.model = null;
        this.zobrist=new ZOBRIST(70000);
        this.engine_buffer=[];
        this.side_buffer=new Array(max_cache);
        this.key_buffer=new Array(max_cache);
        this.pos_cached=0;
        this.n_call=0;
        this.n_evln=0;
        for(var i=0;i<max_cache;i++) 
        {
            this.engine_buffer[i]=[];
            for(var j=0;j<15;j++) 
            {
                this.engine_buffer[i][j]=[];
                for(var k=0;k<15;k++) 
                {
                    this.engine_buffer[i][j][k]=[0,0];
                }
            }
        }
    }
    runEngine()
    {
        this.n_call++;
        this.n_evln+=this.pos_cached;
        console.log("engine called with bs ",this.pos_cached);
        var t0 = performance.now();
        var rst=tf.tidy(() =>{
            return this.model.predict(tf.tensor(this.engine_buffer.slice(0,this.pos_cached))).arraySync();
        });
        var t1 = performance.now();
        engine_t+=(t1-t0);
        return rst;
    }
};

function assign_pending_node(node, policy_logits)
{
    var policy_sum=0;
    for(var idpolicy=0;idpolicy<node.num_child;++idpolicy)
        policy_sum+=policy_logits[node.actions[idpolicy]];
    for(var idpolicy=0;idpolicy<node.num_child;++idpolicy)
        node.children[idpolicy].prior=policy_logits[node.actions[idpolicy]]/policy_sum;
}


function pseudo_evaluate(node,engine)
{
    var wld=winLossDraw();
    if(wld!=-1) return wld;
    if(node.num_child<0) return -2; //evaluate pending
    var policy_logits=null;
    policy_logits=engine.zobrist.getValue(board);
    var nump=0;
    var px=0,py=0;
    var tp=1-node.to_play;
    for(var pos=0;pos<maxPossibleMoves;++pos)
    {
        px=~~(pos/15);py=pos%15;
        if(board[px][py]==0)//since can not be both 1;
        {
            node.actions[nump]=pos;
            node.children[nump]=nodes[nodes_used];
            node.children[nump].set_state(tp);
            ++nodes_used;
            ++nump;
        }
    }
    if(policy_logits!=null)//found in hash
    {
        node.num_child=nump;
        assign_pending_node(node,policy_logits);
        return side2move? 1-policy_logits[maxPossibleMoves] : policy_logits[maxPossibleMoves];
    }
    else//not found
    {
        node.num_child=-nump;//set node to pending state
        engine.key_buffer[engine.pos_cached]=engine.zobrist.currentkey;
        engine.side_buffer[engine.pos_cached]=side2move;
        var npc=engine.pos_cached;
        tp=1-side2move
        // console.log("c",side2move,tp,board[7][7]);
        for(var i=0;i<15;i++) 
        {
            for(var j=0;j<15;j++) 
            {
                switch(board[i][j])
                {
                    case 0:
                        engine.engine_buffer[npc][i][j][0]=engine.engine_buffer[npc][i][j][1]=0;
                        break;
                    case 1:
                        // console.log(1,i,j);
                        engine.engine_buffer[npc][i][j][side2move]=1;
                        engine.engine_buffer[npc][i][j][tp]=0;
                        break;
                    case -1:
                        // console.log(-1,i,j);
                        engine.engine_buffer[npc][i][j][side2move]=0;
                        engine.engine_buffer[npc][i][j][tp]=1;
                        break;
                }
            }
        }
        // printBoard(board);
        // printBoard_bu(engine.engine_buffer[npc]);
        return -1;
    }
}

const fpuReduction=1.3,fpuReductionRoot=1.1;
var pb_c;

function ucb_score(parent,child,isnotroot=true)
{
    var prior_score=pb_c/((child.visit_count+1.0))*(child.prior);
    if(child.visit_count==0)
        return isnotroot? (prior_score-fpuReduction+parent.value()) : (prior_score-fpuReductionRoot+parent.value());
    else
        return prior_score-child.value();
}

function select_child(node,isnotroot=true)
{
    var maxscore=-Infinity,score;
    var maxidx=0;
    pb_c=(Math.log((node.visit_count+19653)/19652.0) + 1.25) * Math.sqrt(node.visit_count);
    for(var i=0;i<node.num_child;i++)
    {
        score=ucb_score(node,node.children[i],isnotroot);
        if(score>maxscore)
        {
            maxscore=score;
            maxidx=i;
        }
    }
    return maxidx;
}

function applyMove(pos,engine)
{
    var px=~~(pos/15),py=pos%15;
    board[px][py]=1-2*side2move;

    engine.zobrist.updatekey(pos*2+side2move);
    side2move=1-side2move;
    movelist[move_count]=pos;
    move_count+=1;
}
function applyMoveXY(px,py,engine){applyMove(px*15+py,engine);}

function takeBack(engine)
{
    move_count-=1;
    var pos=movelist[move_count];
    var px=~~(pos/15),py=pos%15;
    side2move=1-side2move;
    board[px][py]=0;
    engine.zobrist.updatekey(pos*2+side2move);
}

function backpropagate(search_path,lenth,value)
{
    for(var i=0;i<lenth;i++)
    {
        search_path[i].value_sum += search_path[i].to_play == 0? value-1 : (-value);
    }
}

function select_action(root,temperature)
{
    var maxscore=-Infinity,max_id=0;
    var cvcounts=new Array(maxPossibleMoves+1);
    var rnd;
    // console.log("ncl %d\n",root->num_child);
    for(var i=0;i<root.num_child;i++)
    {
        if(root.children[i].visit_count>maxscore) 
        {
            maxscore=root.children[i].visit_count;
            max_id=i;
        }
    }
    if(temperature==0.0)
        return root.actions[max_id];

    var invT=1/temperature
    cvcounts[0]=0.0;
    for(var i=0;i<root.num_child;i++)
    {
        cvcounts[i+1]=((root.children[i].visit_count)<1)? 
                        cvcounts[i]:
                        cvcounts[i]+Math.pow((root.children[i].visit_count)/maxscore,invT);
    }
    rnd=Random.real(0.0,cvcounts[root.num_child],false)(engine);
    for(var i=0;i<root.num_child;i++)
    {
        if(rnd<cvcounts[i+1])
        {
            return root.actions[i];
        }
    }
}

var engine_out_value=0.5,maxdepth=0;

function run_mcts(engine,temperature)
{
    engine.n_call=engine.n_evln=engine_t=0;
    nodes_used=0;
    rootnode.set_state(side2move);
    rootnode.visit_count=1;
    var pseudo_value=pseudo_evaluate(rootnode,engine),eng_output;
    if(pseudo_value>=0)
        rootnode.value_sum=side2move==0? pseudo_value : 1.0-pseudo_value;
        // evaluate returns absolute value: 1 for black wins 0 for white wins
    else
    {
        engine.pos_cached=1;
        eng_output=engine.runEngine()[0];
        rootnode.num_child=-rootnode.num_child;
        assign_pending_node(rootnode,eng_output);
        engine.zobrist.setValue(engine.key_buffer[0],eng_output,engine.engine_buffer[0],engine.side_buffer[0]);
        rootnode.value_sum=eng_output[maxPossibleMoves];
        engine.pos_cached=0;
        // console.log(rootnode.value_sum)
    }
    var idx;
    var depth=new Array(max_cache);
    maxdepth=0;
    var search_path=[],cr_node;
    for(var i=0;i<max_cache;++i)
        search_path[i]=new Array(maxPossibleMoves);
    for(var _t=1;_t<=num_simul;++_t)
    {
        cr_node=rootnode;
        search_path[engine.pos_cached][0]=cr_node;
        depth[engine.pos_cached]=1;

        while(cr_node.expanded())
        {
            idx=select_child(cr_node,depth[engine.pos_cached]>1);
            ++(cr_node.visit_count); ++(cr_node.value_sum);//virtual score to discourage visiting same node//should place after select_child
            applyMove(cr_node.actions[idx],engine);
            cr_node=cr_node.children[idx];
            search_path[engine.pos_cached][depth[engine.pos_cached]++]=cr_node;
        }
        ++(cr_node.visit_count); ++(cr_node.value_sum);//virtual score to discourage visiting same node
        // printBoard();
        pseudo_value=pseudo_evaluate(cr_node,engine);
        // printf("pseudo_value %f %d\n",pseudo_value,engine.pos_cached);
        // printBoard();
        for(var tt=1;tt<depth[engine.pos_cached];++tt) takeBack(engine);
        // printBoard();
        if(pseudo_value>=0)
        {
            backpropagate(search_path[engine.pos_cached],depth[engine.pos_cached],pseudo_value);
            maxdepth=maxdepth>depth[engine.pos_cached]? maxdepth : depth[engine.pos_cached];
        }
        else if(pseudo_value==-2)//leaf is pending
        {
            // printf("revert\n");
            --_t;
            for(var i=0;i<depth[engine.pos_cached];i++)
            {
                --(search_path[engine.pos_cached][i].value_sum);
                --(search_path[engine.pos_cached][i].visit_count);
            }
        }
        else
        {
            ++(engine.pos_cached);
        }
        if(engine.pos_cached==max_cache || pseudo_value==-2 || (_t==num_simul && engine.pos_cached!=0))
        {
            eng_output=engine.runEngine();
            // printf("run finish\n");
            for(var cid=0;cid<engine.pos_cached;++cid)
            {
                search_path[cid][depth[cid]-1].num_child=-search_path[cid][depth[cid]-1].num_child;
                assign_pending_node(search_path[cid][depth[cid]-1],eng_output[cid]);
                engine.zobrist.setValue(engine.key_buffer[cid],eng_output[cid],engine.engine_buffer[cid],engine.side_buffer[cid]);
                // console.log("ev",eng_output[cid][maxPossibleMoves])
                backpropagate(search_path[cid],depth[cid],
                            search_path[cid][depth[cid]-1].to_play? 
                            1-eng_output[cid][maxPossibleMoves] :
                            eng_output[cid][maxPossibleMoves]);
                maxdepth=maxdepth>depth[cid]? maxdepth : depth[cid];
                // printf("root v %f\n",rootnode.value_sum);
            }
            engine.pos_cached=0;
        }
        // console.log(rootnode.value_sum)
    }
    engine_out_value=rootnode.value();
    console.log("predict_call:",engine.n_evln,"|",engine.n_call,"max_depth:",maxdepth);
    return select_action(rootnode,temperature);
}

function clearAll(engine)
{
    move_count=0;
    side2move=0;
    for(var px=0;px<15;px++)
        for(var py=0;py<15;py++)
        {
            board[px][py]=0;
        }
    engine.zobrist.clearkey();
}
