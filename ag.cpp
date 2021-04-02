#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <random>
#include <cmath>
#include <chrono>
#include "tensorflow/c/c_api.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

float board[15][15][2],inv_board[15][15][2];
std::mt19937 mt_19937;

const int maxPossibleMoves=15*15;
auto side2move=0;
int movelist[maxPossibleMoves];
auto move_count=0;
auto num_simul=800;
auto const sfmxMoves=30;
const int max_cache=4;
// FILE *gammadis,*uniformdis;

void printBoard(float* mboard=board[0][0])
{
    printf("   ");
    for(auto ii=0;ii<15;ii++)
        printf(" %d ",ii%10);
    printf("\n");
    for(auto ii=0;ii<15;ii++)
    {
        printf(" %d ",ii%10);
        for(auto jj=0;jj<15;jj++)
        {
            if(mboard[(ii*15+jj)*2+0]==0.f and mboard[(ii*15+jj)*2+1]==0.f)
            {
                if((ii==7 and jj==7) or ((ii==3 or ii==11) and (jj==3 or jj==11)))
                    printf(" + ");
                else
                    printf(" - ");
            }
            else if(mboard[(ii*15+jj)*2+0]==1.f and mboard[(ii*15+jj)*2+1]==0.f)
                printf("\033[96m O \033[0m");
            else if(mboard[(ii*15+jj)*2+0]==0.f and mboard[(ii*15+jj)*2+1]==1.f)
                printf("\033[91m X \033[0m");
            else
                printf(" E ");
        }
        printf(" %d\n",ii%10);
    }
    printf("   ");
    for(auto ii=0;ii<15;ii++)
        printf(" %d ",ii%10);
    printf("\n");
}

class NODE
{
public:
    int visit_count=0;
    int to_play;
    float prior;
    float value_sum=0.0;
    int actions[maxPossibleMoves];
    NODE* children[maxPossibleMoves];
    int num_child=0;

    void set_state(int sidem);
    bool expanded();
    float value();
};
void NODE::set_state(int sidem)
{
    visit_count=0;
    to_play=sidem;
    value_sum=0.0;
    num_child=0;
}
bool NODE::expanded()
{
    return (num_child>0);
}
float NODE::value()
{
    //unvisited node call result in error!!!
    return value_sum/visit_count;
}

int node_capacity=(num_simul+1)*224>500? (num_simul+1)*224 : 500;
NODE *nodes=new NODE[node_capacity];

void setNum_sml(int nsml)
{
    int sz;
    num_simul=nsml;
    if((sz=(num_simul+1)*224)>node_capacity)
    {
        node_capacity=sz;
        delete[] nodes;
        nodes=new NODE[node_capacity];
    }
    printf("num_simul = %d\n",num_simul);
}

NODE rootnode;
auto nodes_used=0;

float winLossDraw()//return O wins or X wins
{
    auto sd=1-side2move;
    auto posx=movelist[move_count-1]/15, posy=movelist[move_count-1]%15;

    auto lx=1,lg=1;
    while(posx-lx>=0 and board[posx-lx][posy][sd]==1.f){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 and board[posx+lx][posy][sd]==1.f){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posy-lx>=0 and board[posx][posy-lx][sd]==1.f){lg+=1; lx+=1;}
    lx=1;
    while(posy+lx<15 and board[posx][posy+lx][sd]==1.f){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posx-lx>=0 and posy-lx>=0 and board[posx-lx][posy-lx][sd]==1.f){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 and posy+lx<15 and board[posx+lx][posy+lx][sd]==1.f){lg+=1; lx+=1;}
    if(lg>=5) return side2move;

    lx=1;lg=1;
    while(posx-lx>=0 and posy+lx<15 and board[posx-lx][posy+lx][sd]==1.f){lg+=1; lx+=1;}
    lx=1;
    while(posx+lx<15 and posy-lx>=0 and board[posx+lx][posy-lx][sd]==1.f){lg+=1; lx+=1;}
    if(lg>=5) return side2move;
    
    if(move_count==maxPossibleMoves) return 0.5;
    return -1;
}

class ZOBRIST
{
public:
    ZOBRIST(const int sze);
    int hash_size,table[2*maxPossibleMoves],initkey,currentkey;
    float *values,*boardcheck;
    float* getValue(float *boardX);
    void setValue(float *rst_in, float *board_in);
    void setValue(int key, float *rst_in, float *board_in);
    void updatekey(int pos);
    void clearkey();
};

void ZOBRIST::updatekey(int pos)
{
    currentkey^=table[pos];
}


void ZOBRIST::clearkey()
{
    currentkey=initkey;
}
ZOBRIST::ZOBRIST(const int sze)
{
    auto lg=1;
    while(sze>>lg) lg++;
    hash_size=1<<(lg-1);
    printf("Hash size = %d\n",hash_size);
    for(auto i=0;i<2*maxPossibleMoves;i++)
        table[i]=mt_19937()%hash_size;
    currentkey=initkey=mt_19937()%hash_size;
    values=new float[(maxPossibleMoves+1)*hash_size];
    boardcheck=new float[2*maxPossibleMoves*hash_size];
    for(auto i=0;i<2*maxPossibleMoves*hash_size;i++)
    boardcheck[i]=-1;
}
float* ZOBRIST::getValue(float *boardX)
{
    // int key=initkey;
    // for(auto ii=0;ii<2*maxPossibleMoves;ii++)
    // {
        // if(boardX[ii]==1.f) key^=table[ii];
    // }
    // *key_out=key;
    // if(key!=currentkey)
        // printf("WARNING: bat key: %d | %d\n\n",key,currentkey);
    if(memcmp(boardX,&boardcheck[currentkey*2*maxPossibleMoves],sizeof(board)) == 0)
        return &values[currentkey*(maxPossibleMoves+1)];
    else
        return NULL;

}

void ZOBRIST::setValue(float *rst_in, float *board_in)
{
    memcpy(&values[currentkey*(maxPossibleMoves+1)],rst_in,(maxPossibleMoves+1)*sizeof(float));
    memcpy(&boardcheck[currentkey*2*maxPossibleMoves],board_in,(maxPossibleMoves*2)*sizeof(float));
}
void ZOBRIST::setValue(int key, float *rst_in, float *board_in)
{
    memcpy(&values[key*(maxPossibleMoves+1)],rst_in,(maxPossibleMoves+1)*sizeof(float));
    memcpy(&boardcheck[key*2*maxPossibleMoves],board_in,(maxPossibleMoves*2)*sizeof(float));
}

class A0ENGINE
{
public:
    TF_Session* Session;
    TF_Output* Input;
    TF_Output* Output;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    const int NumInputs = 1;
    const int NumOutputs = 1;
    int pos_cached=0;
    TF_Status* Status;
    float* engBoard[max_cache];
    int initEngine(const char* saved_model_dir);
    void runEngine();
    ZOBRIST *zobrist;
    int z_key[max_cache];
    float eboard[max_cache*15*15*2];
};

int A0ENGINE::initEngine(const char* saved_model_dir)
{
    //********* Read model
    TF_Graph* Graph = TF_NewGraph();
    Status = TF_NewStatus();

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;
    // uint8_t intra_op_parallelism_threads = 1;
    // uint8_t inter_op_parallelism_threads = 1;
    // uint8_t device_count = 1;//device_count limits the number of CPUs.**strong text**
    // uint8_t config[15] = {0xa, 0x7, 0xa, 0x3, 0x43, 0x50, 0x55, 0x10, device_count, 0x10, intra_op_parallelism_threads, 0x28, intra_op_parallelism_threads,0x40, 0x1};
    // TF_SetConfig(SessionOpts, (void*)config, 13, Status);
    uint8_t config[] = {0x10, 0x1, 0x28, 0x1};
    TF_SetConfig(SessionOpts, (void*)config, 4, Status);

    if (TF_GetCode(Status)!= TF_OK)
        printf("%s",TF_Message(Status));

    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;

    Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if(TF_GetCode(Status) == TF_OK)
    {
        printf("TF_LoadSessionFromSavedModel OK\n");
    }
    else
    {
        printf("%s",TF_Message(Status));
        return 1;
    }
    //****** Get input tensor
    //TODO : need to use saved_model_cli to read saved_model arch
    Input = new TF_Output[NumInputs];

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0;
    
    //********* Get Output tensor
    Output = new TF_Output[NumOutputs];

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    InputValues =  new TF_Tensor*[max_cache*NumInputs ];
    OutputValues = new TF_Tensor*[max_cache*NumOutputs];

    int ndims = 4;
    int64_t dims[] = {1,15,15,2};
    int ndata = sizeof(float)*450 ;// This is tricky, it number of bytes not number of element

    printf("eboard dict: %lu\n",(unsigned long)eboard);
    for(auto ix=0;ix<max_cache;++ix)
    {
        dims[0]=ix+1; ndata=sizeof(float)*450*(ix+1);
        InputValues[ix]=TF_NewTensor(TF_FLOAT,dims,ndims,eboard,ndata,&NoOpDeallocator,0);
        if (InputValues[ix] != NULL)
        {
            engBoard[ix]=(float*)TF_TensorData(InputValues[ix]);
            printf("TF_NewTensor is OK: %lu\n",(unsigned long)engBoard[ix]);
        }
        else
            printf("ERROR: Failed TF_NewTensor\n");
    }

    zobrist=new ZOBRIST(500000);
    return 0;
}

void A0ENGINE::runEngine()
{
    TF_SessionRun(Session, NULL, Input, &InputValues[pos_cached-1], NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
    // if(TF_GetCode(Status) != TF_OK) printf("%s",TF_Message(Status));
}

void assign_pending_node(NODE* node, float* policy_logits)
{
    float policy_sum=0;
    // printf("chld count: %d\n",node->num_child);
    for(auto idpolicy=0;idpolicy<node->num_child;++idpolicy)
        policy_sum+=policy_logits[node->actions[idpolicy]];
    for(auto idpolicy=0;idpolicy<node->num_child;++idpolicy)
        node->children[idpolicy]->prior=policy_logits[node->actions[idpolicy]]/policy_sum;
}

float pseudo_evaluate(NODE* node,A0ENGINE* engine,int cache_pos=max_cache-1)
{
    auto wld=winLossDraw();
    if(wld!=-1.f) return wld;
    if(node->num_child<0) return -2; //evaluate pending
    float* policy_logits=NULL;
    // policy_logits=NULL;
    policy_logits=engine->zobrist->getValue(side2move? inv_board[0][0]:board[0][0]);//hash table stores absolute positions: [black,white] not [self,opponent].
    auto nump=0;
    auto px=0,py=0;
    auto tp=1-node->to_play;
    for(auto pos=0;pos<maxPossibleMoves;pos++)
    {
        px=pos/15;py=pos%15;
        if(board[px][py][0]==board[px][py][1])//since can not be both 1;
        {
            node->actions[nump]=pos;
            node->children[nump]= &nodes[nodes_used];
            node->children[nump]->set_state(tp);
            ++nodes_used;
            ++nump;
        }
    }
    if(policy_logits)//found in hash
    {
        node->num_child=nump;
        assign_pending_node(node,policy_logits);
        return side2move? 1-policy_logits[maxPossibleMoves] : policy_logits[maxPossibleMoves];
    }
    else//not found
    {
        node->num_child=-nump;//set node to pending state
        if(side2move) memcpy(&(engine->engBoard[cache_pos][450*engine->pos_cached]),inv_board[0][0],sizeof(inv_board));
        else          memcpy(&(engine->engBoard[cache_pos][450*engine->pos_cached]),board[0][0],sizeof(board));
        engine->z_key[engine->pos_cached]=engine->zobrist->currentkey;
        return -1;
    }
}

std::gamma_distribution<float> GammaDistribution(.05,1.0);
std::uniform_real_distribution<float> UniformDistribution(0.0,1.0);

void add_exploration_noise(NODE *node)
{
    float noise[maxPossibleMoves],noise_sum=0;
    for(auto i=0;i<node->num_child;i++)
    {
        noise[i]=GammaDistribution(mt_19937);
        // fscanf(gammadis,"%f\n",&noise[i]);
        noise_sum+=noise[i];
    }
    for(auto i=0;i<node->num_child;i++)
    {
        node->children[i]->prior = node->children[i]->prior*.75+noise[i]/noise_sum*.25;
    }
}

float fpuReduction=1.3f,fpuReductionRoot=1.0f;
float pb_c,cpuct=1.25f;

float ucb_score(NODE *parent, NODE *child, bool isnotroot=true)
{
    float prior_score=pb_c/((float)(child->visit_count+1))*(child->prior);
    if(child->visit_count==0)
        return isnotroot? (prior_score-fpuReduction+parent->value()) : (prior_score-fpuReductionRoot+parent->value());
    else
        return prior_score-child->value();
}

int select_child(NODE *node,bool isnotroot=true)
{
    // return mt_19937()%node->num_child;
    float maxscore=-INFINITY,score;
    int maxidx=0;
    pb_c=(logf((node->visit_count+19653)/19652.0f)+cpuct) * sqrtf(node->visit_count);
    for(auto i=0;i<node->num_child;i++)
    {
        score=ucb_score(node,node->children[i],isnotroot);
        if(score>maxscore)
        {
            maxscore=score;
            maxidx=i;
        }
    }
    return maxidx;
}

void applyMove(int pos,ZOBRIST *hzobrist)
{
    auto px=pos/15,py=pos%15;
    board[px][py][side2move]=inv_board[px][py][1-side2move]=1;
    hzobrist->updatekey(pos*2+side2move);
    side2move=1-side2move;
    movelist[move_count]=pos;
    move_count+=1;
}
inline void applyMove(int px,int py,ZOBRIST *hzobrist){applyMove(px*15+py,hzobrist);}

void takeBack(ZOBRIST *hzobrist)
{
    move_count-=1;
    auto pos=movelist[move_count];
    auto px=pos/15,py=pos%15;
    side2move=1-side2move;
    board[px][py][side2move]=inv_board[px][py][1-side2move]=0;
    hzobrist->updatekey(pos*2+side2move);
}

void backpropagate(NODE **search_path,int lenth,float value)
{
    for(auto i=0;i<lenth;i++)
        search_path[i]->value_sum += search_path[i]->to_play == 0? value-1 : (-value);
}

float valueWt=0*num_simul;
float actionScore(NODE *node)
{
    if(node->visit_count<1)
        return -valueWt;
    else
        return node->visit_count-valueWt*node->value();
        //range:(-valueWt,num_simul)
}

int select_action(NODE *root,bool add_noise=true)
{
    float maxscore=-INFINITY,cscore;
    int action=0;
    float cvcounts[maxPossibleMoves+1];
    float rnd;
    if(move_count>sfmxMoves or (not add_noise))
    {
        for(auto i=0;i<root->num_child;i++)
        {
            cscore=actionScore(root->children[i]);
            // printf("%d %f\n",i,cscore);
            if(cscore>maxscore)
            {
                maxscore=cscore;
                action=root->actions[i];
            }
        }
    }
    else
    {
        // printf("ncl %d\n",root->num_child);
        for(auto i=0;i<root->num_child;i++)
            if(root->children[i]->visit_count>maxscore) maxscore=root->children[i]->visit_count;
        cvcounts[0]=0.f;
        for(auto i=0;i<root->num_child;i++)
        {
            // cvcounts[i+1]=cvcounts[i]+powf((root->children[i]->visit_count)/maxscore,1.5f);
            cvcounts[i+1]=((root->children[i]->visit_count)<3)? 
                            cvcounts[i]:
                            cvcounts[i]+(root->children[i]->visit_count)/maxscore;
        }
        rnd=UniformDistribution(mt_19937)*cvcounts[root->num_child];
        // fscanf(uniformdis,"%f\n",&rnd);
        // rnd*=cvcounts[root->num_child];
        // action=root->actions[(root->num_child)-1];
        // printf("%f %f\n",cvcounts[root->num_child],rnd);
        for(auto i=0;i<root->num_child;i++)
        {
            if(rnd<cvcounts[i+1])
            {
                action=root->actions[i];
                break;
            }
        }
    }
    return action;
}

int run_mcts(A0ENGINE* engine,bool sel_noise,bool dir_noise,float *prb_out,float *value_out, int* depth_out,float* time_out)
{
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    nodes_used=0;
    rootnode.set_state(side2move);
    rootnode.visit_count=1;
    float pseudo_value=pseudo_evaluate(&rootnode,engine,0),*eng_output;
    if(pseudo_value>=0.f)
        rootnode.value_sum=side2move==0? pseudo_value : 1.0f-pseudo_value;
        // evaluate returns absolute value: 1 for black wins 0 for white wins
    else
    {
        ++(engine->pos_cached);
        engine->runEngine();
        eng_output=(float*)TF_TensorData(engine->OutputValues[0]);
        rootnode.num_child=-rootnode.num_child;
        assign_pending_node(&rootnode,eng_output);
        engine->zobrist->setValue(engine->z_key[0],eng_output,board[0][0]);
        rootnode.value_sum=side2move==0? eng_output[maxPossibleMoves] : 1.0f-eng_output[maxPossibleMoves];
        TF_DeleteTensor(engine->OutputValues[0]);
        engine->pos_cached=0;
    }
    // printf("root v %f\n",rootnode.value_sum);
    if(dir_noise) add_exploration_noise(&rootnode);
    // printf("noise!\n");
    int idx;
    int depth[max_cache],maxdepth=0;
    NODE *search_path[max_cache][maxPossibleMoves],*cr_node;
    for(auto _t=1;_t<=num_simul;++_t)
    {
        cr_node=&rootnode;
        search_path[engine->pos_cached][0]=cr_node;
        depth[engine->pos_cached]=1;

        while(cr_node->expanded())
        {
            idx=select_child(cr_node,depth[engine->pos_cached]>1);
            // printf("d %d ",depth[engine->pos_cached]);
            // printf("select chld %d\n",idx);
            ++(cr_node->visit_count); ++(cr_node->value_sum);//virtual score to discourage visiting same node//should place after select_child
            applyMove(cr_node->actions[idx],engine->zobrist);
            cr_node=cr_node->children[idx];
            search_path[engine->pos_cached][depth[engine->pos_cached]++]=cr_node;
        }
        ++(cr_node->visit_count); ++(cr_node->value_sum);//virtual score to discourage visiting same node
        // printBoard();
        pseudo_value=pseudo_evaluate(cr_node,engine);
        // printf("pseudo_value %f %d\n",pseudo_value,engine->pos_cached);
        // printBoard();
        for(auto tt=1;tt<depth[engine->pos_cached];++tt) takeBack(engine->zobrist);
        // printBoard();
        if(pseudo_value>=0)
        {
            backpropagate(search_path[engine->pos_cached],depth[engine->pos_cached],pseudo_value);
            // printf("root v %f\n",rootnode.value_sum);
            maxdepth=maxdepth>depth[engine->pos_cached]? maxdepth : depth[engine->pos_cached];
        }
        else if(pseudo_value==-2)//leaf is pending
        {
            // printf("revert\n");
            --_t;
            for(auto i=0;i<depth[engine->pos_cached];i++)
            {
                --(search_path[engine->pos_cached][i]->value_sum);
                --(search_path[engine->pos_cached][i]->visit_count);
            }
        }
        else
        {
            ++(engine->pos_cached);
        }
        if(engine->pos_cached==max_cache || pseudo_value==-2 || (_t==num_simul && engine->pos_cached!=0))
        {
            // printf("run\n");
            if(engine->pos_cached!=max_cache)
            {
                // printf("data moved\n");
                memcpy(engine->engBoard[engine->pos_cached-1],engine->engBoard[max_cache-1],sizeof(float)*450*engine->pos_cached);
            }
            engine->runEngine();
            eng_output=(float*)TF_TensorData(engine->OutputValues[0]);
            // printf("run finish\n");
            for(auto cid=0;cid<engine->pos_cached;++cid)
            {
                search_path[cid][depth[cid]-1]->num_child=-search_path[cid][depth[cid]-1]->num_child;
                assign_pending_node(search_path[cid][depth[cid]-1],&(eng_output[cid*(maxPossibleMoves+1)]));
                // printBoard(&(engine->engBoard[max_cache-1][maxPossibleMoves*2*cid]));
                // printf("engine %f %d\n",eng_output[cid*(maxPossibleMoves+1)+maxPossibleMoves],engine->z_key[cid]);
                engine->zobrist->setValue(engine->z_key[cid],&eng_output[cid*(maxPossibleMoves+1)],&(engine->engBoard[max_cache-1][maxPossibleMoves*2*cid]));
                backpropagate(search_path[cid],depth[cid],
                            search_path[cid][depth[cid]-1]->to_play? 
                            1-eng_output[cid*(maxPossibleMoves+1)+maxPossibleMoves] :
                            eng_output[cid*(maxPossibleMoves+1)+maxPossibleMoves]);
                maxdepth=maxdepth>depth[cid]? maxdepth : depth[cid];
                // printf("root v %f\n",rootnode.value_sum);
            }
            TF_DeleteTensor(engine->OutputValues[0]);
            engine->pos_cached=0;
        }
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<float> time_span = duration_cast<duration<float> >(t2 - t1);
    if(prb_out)
    {
        for(auto i=0;i<maxPossibleMoves;i++) prb_out[i]=0;
        for(auto i=0;i<rootnode.num_child;i++)
            prb_out[rootnode.actions[i]]=float(rootnode.children[i]->visit_count)/num_simul;
    }
    if(value_out) *value_out=rootnode.value();
    if(depth_out) *depth_out=maxdepth;
    if(time_out) *time_out=time_span.count();
    return select_action(&rootnode,sel_noise);
}

void clearAll(ZOBRIST *hzobrist)
{
    move_count=0;
    side2move=0;
    for(auto px=0;px<15;px++)
        for(auto py=0;py<15;py++)
            board[px][py][0]=inv_board[px][py][0]=board[px][py][1]=inv_board[px][py][1]=0;
    hzobrist->clearkey();
}

int partition(int* lst,int lo,int hi)//[lo,hi]
{
    int pivot=rootnode.children[lst[hi]]->visit_count,i=lo,temp;
    for(auto j=lo;j<=hi;j++)
    {
        if(rootnode.children[lst[j]]->visit_count<pivot)
        {
            temp=lst[i];lst[i]=lst[j];lst[j]=temp;
            ++i;
        }
    }
    temp=lst[i];lst[i]=lst[hi];lst[hi]=temp;
    return i;
}
void quick_sort(int* lst,int lo,int hi)
{
    if(lo<hi)
    {
        auto p=partition(lst,lo,hi);
        quick_sort(lst,lo,p-1);
        quick_sort(lst,p+1,hi);
    }
}

void hmpl()
{
    A0ENGINE a0eng;
    a0eng.initEngine("RNG_Old/");
    fpuReductionRoot=1.3f;
    float val,tmu;
    int act,dpt,mvs[maxPossibleMoves];
    char cmd;
    int px,py;
    while(true)
    {
        // printf("input cmd > ");
        fflush(stdout);
        scanf("%c",&cmd);
        // printf("cmd is %c(%d)\n",cmd,cmd);
        switch (cmd)
        {
        case 'p':
            if(scanf("%d %d",&px,&py)!=2) {printf("Invalid command\n"); break;}
            printf("Play %d %d\n",px,py);
            if(0<=px and px<15 and 0<=py and py<15 and board[px][py][0]==board[px][py][0]) applyMove(px,py,a0eng.zobrist);
            else printf("Illegal\n");
            printBoard();
            if(winLossDraw()>-.5f) printf("Game over\n");
            break;
        case 'v':
            printf("Comp play ");
            act=run_mcts(&a0eng,false,false,NULL,&val,&dpt,&tmu);
            printf("%3d %2d %2d %.3f %3d %.3f\n",move_count,act/15,act%15,val,dpt,tmu);
            // applyMove(act);
            // printBoard();
            // if(winLossDraw()>-.5f) printf("Game over\n");
            break;
        case 't':
            printf("Comp play ");
            act=run_mcts(&a0eng,false,true,NULL,&val,&dpt,&tmu);
            for(auto km=0;km<rootnode.num_child;++km)
                mvs[km]=km;
            quick_sort(mvs,0,rootnode.num_child-1);
            for(auto kz=0;kz<rootnode.num_child;++kz)
            {
                act=mvs[kz];
                printf("(%-2d,%-2d) pr %.3f nv %-3d sc %.3f\n",rootnode.actions[act]/15,rootnode.actions[act]%15,rootnode.children[act]->prior,rootnode.children[act]->visit_count,1-rootnode.children[act]->value());
            }
            break;
        case 'n':
            if(scanf("%d",&px)!=1) printf("Invalid command\n");
            else setNum_sml(px);
            break;
        case 'f':
            if(scanf("%f %f",&fpuReduction,&fpuReductionRoot)!=2) printf("Invalid command\n");
            printf("FPU = %f | %f\n",fpuReduction,fpuReductionRoot);
            break;
        case 'u':
            if(scanf("%f",&cpuct)!=1) printf("Invalid command\n");
            printf("cpuct = %f\n",cpuct);
            break;
        case 'w':
            if(scanf("%f",&valueWt)!=1) printf("Invalid command\n");
            printf("valueWt = %f | %f\n",valueWt,valueWt*num_simul);
            valueWt=valueWt*num_simul;
            break;
        case 'b':
            if(move_count==0) printf("Cannot take back\n");
            else{printf("Take back\n");takeBack(a0eng.zobrist);}
            break;
        case 'c':
            printf("New game\n");
            clearAll(a0eng.zobrist);
            break;
        case 'q':
            printf("Quit\n");
            return;
            break;
        default:
            printf("Invalid command %c(%d)\n",cmd,cmd);
            break;
        }
        while((cmd = getchar()) != '\n' && cmd != EOF);
        // ungetc('n',stdin);
        printf("task completed\n");
        fflush(stdout);
    }
}
const int canonical_op[][2]={
    {5,9},{6,9},{7,9},{8,9},{9,9},
                {7,8},{8,8},{9,8},
                      {8,7},{9,7},
                      {8,6},{9,6},
                            {9,5},
                {5,7},{5,8},{5,9},
                      {6,8},{6,9},
                      {7,8},{7,9},
                {8,7},{8,8},{8,9},
                {9,7},{9,8},{9,9},
};
std::uniform_int_distribution<int> rand_int_op(0,25);
std::uniform_int_distribution<int> rand_int_ps(0,14);
std::uniform_int_distribution<int> rand_int_dp(-2,2);
int rnd_opening(int *ops)
{
    float ds=UniformDistribution(mt_19937);
    int n_op;
    if(ds<.05)
    {
        return 0;
    }
    else if(ds<.3)
    {
        ops[0]=rand_int_ps(mt_19937);
        ops[1]=rand_int_ps(mt_19937);
        return 1;
    }
    else
    {
        ops[0]=ops[1]=7;
        n_op=rand_int_op(mt_19937);
        printf("opening #%d\n",n_op);
        ops[2]=6;
        ops[3]=n_op<13? 8 : 7;
        ops[4]=canonical_op[n_op][0];
        ops[5]=canonical_op[n_op][1];
        if(UniformDistribution(mt_19937)<.5)
        {
            // printf("Transpose xy\n");
            for(auto kk=0;kk<3;kk++) {n_op=ops[kk*2+0];ops[kk*2+0]=ops[kk*2+1];ops[kk*2+1]=n_op;}
        }
        if(UniformDistribution(mt_19937)<.5)
        {
            // printf("Flip x\n");
            for(auto kk=0;kk<3;kk++) {ops[kk*2+0]=14-ops[kk*2+0];}
        }
        if(UniformDistribution(mt_19937)<.5)
        {
            // printf("Flip y\n");
            for(auto kk=0;kk<3;kk++) {ops[kk*2+1]=14-ops[kk*2+1];}
        }
        // printf("x-shift: %2d\n",(act=mt_19937()%3-1));
        n_op=rand_int_dp(mt_19937);
        for(auto kk=0;kk<3;kk++) {ops[kk*2+0]+=n_op;}
        // printf("y-shift: %2d\n",(act=mt_19937()%3-1));
        n_op=rand_int_dp(mt_19937);
        for(auto kk=0;kk<3;kk++) {ops[kk*2+1]+=n_op;}
        return 3;
    }
}

void sfpl(int npos,const char* out_file,int rseed)
{
    mt_19937.seed(rseed);
    A0ENGINE a0eng;
    a0eng.initEngine("RNG_Old/");
    float val,tmu;
    int act,dpt;
    float *bufx=new float[maxPossibleMoves*maxPossibleMoves*2],*bufy=new float[maxPossibleMoves*(maxPossibleMoves+1)];
    char ff[128];
    float game_rst=-1;
    int totps=0;
    int ngames=0;
    float gs_sum=0;
    int opns[3][2];
    int nops=0;
    FILE *fpx,*fpy;
    sprintf(ff,"games/%s.x",out_file);
    fpx=fopen(ff,"a+b");
    if(!fpx){printf("error! %s\n",ff);return;}
    sprintf(ff,"games/%s.y",out_file);
    fpy=fopen(ff,"a+b");
    if(!fpy){printf("error! %s\n",ff);return;}
    fpuReduction=1.3f; fpuReductionRoot=1.0f;
    cpuct=2.5f;
    printf("fpu = %.3f | %.3f\n",fpuReduction,fpuReductionRoot);
    printf("cpunt = %.3f\n",cpuct);

    // gammadis=fopen("gamma.txt","r");
    // uniformdis=fopen("uniform.txt","r");

    while(totps<npos)
    {
        printf("game %d\n",ngames+1);
        clearAll(a0eng.zobrist);
        nops=rnd_opening(opns[0]);
        for(auto kk=0;kk<nops;kk++)
        {
            applyMove(opns[kk][0],opns[kk][1],a0eng.zobrist);
        }
        while(move_count==0 or (game_rst=winLossDraw())< -.5)
        {
            if(side2move) memcpy(&bufx[move_count*(maxPossibleMoves*2)],inv_board[0][0],sizeof(inv_board));
            else memcpy(&bufx[move_count*(maxPossibleMoves*2)],board[0][0],sizeof(board));
            // printf("copy complete\n");
            act=run_mcts(&a0eng,true,true,&bufy[move_count*(maxPossibleMoves+1)],&val,&dpt,&tmu);
            bufy[move_count*(maxPossibleMoves+1)+maxPossibleMoves]=val;//use node val;
            printf("\r%3d %2d %2d %.3f %3d %.3f ",move_count,act/15,act%15,val,dpt,tmu);
            fflush(stdout);
            applyMove(act,a0eng.zobrist);
        }
        // for(auto kk=nops;kk<move_count;kk++)//use game result for val
        // {
        //     bufy[kk*(maxPossibleMoves+1)+maxPossibleMoves]=kk%2? 
        //                                                     .9*bufy[kk*(maxPossibleMoves+1)+maxPossibleMoves]+.1*(1-game_rst) :
        //                                                     .9*bufy[kk*(maxPossibleMoves+1)+maxPossibleMoves]+.1*(game_rst  ) ;
        //     //game_rst is absolute result!
        // }
        ngames++;
        gs_sum+=game_rst;
        printf("\r");
        printBoard();
        totps+=(move_count-nops);
        printf("result: %.1f | %d, %d games, %d pos, avg_score_black: %.3f\n",game_rst,nops,ngames,totps,gs_sum/ngames);
        fwrite(&bufx[nops*(maxPossibleMoves*2)],sizeof(float),(move_count-nops)*(maxPossibleMoves*2),fpx);
        fwrite(&bufy[nops*(maxPossibleMoves+1)],sizeof(float),(move_count-nops)*(maxPossibleMoves+1),fpy);
    }
    printf("generated %d pos\n",totps);
    fclose(fpx);
    fclose(fpy);
    delete[] bufx;
    delete[] bufy;
    // fclose(gammadis);
    // fclose(uniformdis);
}

void sfvs(const char* out_file,const char* opening,int rseed)
{
    mt_19937.seed(rseed);
    A0ENGINE a0eng1;
    A0ENGINE a0eng2;
    float val,tmu;
    int act,dpt;
    char ff[128];
    float game_rst=-1;
    int ngames=0;
    int opns[3][2];
    int nops=0;
    auto side_id=0;
    float nscore=0;
    FILE *fp;
    setNum_sml(800);
    fpuReduction=fpuReductionRoot=1.3f;
    a0eng1.initEngine("RNG_Old/");
    a0eng2.initEngine("RNG/");

    sprintf(ff,"eval/%s.txt",opening);
    fp=fopen(ff,"r");
    if(!fp){printf("error! %s\n",ff);return;}
    nops=fscanf(fp,"%d %d %d %d %d %d",&opns[0][0],&opns[0][1],&opns[1][0],&opns[1][1],&opns[2][0],&opns[2][1])/2;
    fclose(fp);
    printf("%d opening moves:\n",nops);
    for(auto kk=0;kk<nops;kk++) printf("%d (%-2d,%-2d)\n",kk,opns[kk][0],opns[kk][1]);
    if(UniformDistribution(mt_19937)<.5)
    {
        printf("Transpose xy\n");
        for(auto kk=0;kk<nops;kk++) {act=opns[kk][0];opns[kk][0]=opns[kk][1];opns[kk][1]=act;}
    }
    if(UniformDistribution(mt_19937)<.5)
    {
        printf("Flip x\n");
        for(auto kk=0;kk<nops;kk++) {opns[kk][0]=14-opns[kk][0];}
    }
    if(UniformDistribution(mt_19937)<.5)
    {
        printf("Flip y\n");
        for(auto kk=0;kk<nops;kk++) {opns[kk][1]=14-opns[kk][1];}
    }
    // printf("x-shift: %2d\n",(act=mt_19937()%3-1));
    // for(auto kk=0;kk<nops;kk++) {opns[kk][0]+=act;}
    // printf("y-shift: %2d\n",(act=mt_19937()%3-1));
    // for(auto kk=0;kk<nops;kk++) {opns[kk][1]+=act;}

    sprintf(ff,"eval/%s.txt",out_file);
    fp=fopen(ff,"w");
    if(!fp){printf("error! %s\n",ff);return;}
    for(ngames=0;ngames<2;ngames++)
    {
        printf("game %d\n",ngames+1);
        a0eng2.zobrist->clearkey();
        clearAll(a0eng1.zobrist);
        for(auto kk=0;kk<nops;kk++) 
        {
            fprintf(fp,"%3d b %2d %2d\n",move_count,opns[kk][0],opns[kk][1]);
            a0eng2.zobrist->updatekey((opns[kk][0]*15+opns[kk][1])*2+side2move);//updatekey must be called before applyMove!
            applyMove(opns[kk][0],opns[kk][1],a0eng1.zobrist);
        }
        while(move_count==0 or (game_rst=winLossDraw())< -.5)
        {
            side_id=(ngames+move_count)%2;
            if(side_id==0) act=run_mcts(&a0eng1,false,false,NULL,&val,&dpt,&tmu);
            else act=run_mcts(&a0eng2,false,false,NULL,&val,&dpt,&tmu);
            printf("\r%3d %d %2d %2d %.3f %3d %.3f ",move_count,side_id,act/15,act%15,val,dpt,tmu);
            fflush(stdout);
            fprintf(fp,"%3d %d %2d %2d %.3f %3d %.3f\n",move_count,side_id,act/15,act%15,val,dpt,tmu);
            a0eng2.zobrist->updatekey(act*2+side2move);//updatekey must be called before applyMove!
            applyMove(act,a0eng1.zobrist);
        }
        printf("\r");
        printBoard();
        if(game_rst==1.0f or game_rst==0.0f)
            nscore+=side_id;
        else
            nscore+=.5;
        printf("result: %.1f | %.1f\n",game_rst,nscore);
        fprintf(fp,"result: %.1f\n",game_rst);
    }
    fclose(fp);
    sprintf(ff,"eval/%sr.txt",out_file);
    fp=fopen(ff,"w");
    if(!fp){printf("error! %s\n",ff);return;}
    fprintf(fp,"%.1f\n",nscore);
    fclose(fp);
}

int main(int argc, const char* argv[])
{
    const char *buildString = "Ag by YZ, compiled at " __DATE__ ", " __TIME__ ".\n";
    printf(buildString);
    const char errormsg[]="usage:\n  h                                   ->   hmpl\n  t <n_pos> <out_file> <seed>         ->   self play\n  v <out_file> <opening_file> <seed>  ->   versus\n";
    int sd,np;
    if(argc==1) 
    {
        printf(errormsg);
        return 1;
    }
    else if(argv[1][0]=='h')
    {
        printf("hmpl\n");
        hmpl(); 
    }
    else if(argv[1][0]=='t')
    {
        if(argc!=5)
        {
            printf(errormsg);
           return 1;
        }
        sscanf(argv[2],"%d",&np);
        sscanf(argv[4],"%d",&sd);
        printf("n_pos = %d\nfile = %s.x|y\nseed = %d\n",np,argv[3],sd);
        sfpl(np,argv[3],sd);
    }
    else if(argv[1][0]=='v')
    {
        if(argc!=5)
        {
            printf(errormsg);
           return 1;
        }
        sscanf(argv[4],"%d",&sd);
        printf("op = %s\nfile = %s.x|y\nseed = %d\n",argv[2],argv[3],sd);
        sfvs(argv[2],argv[3],sd);
    }
    else
    {
        printf(errormsg);
        return 1;
    }
    return 0;
}