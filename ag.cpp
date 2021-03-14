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
auto sfmxMoves=30;

void printBoard()
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
            if(board[ii][jj][0]==0.f and board[ii][jj][1]==0.f)
            {
                if((ii==7 and jj==7) or ((ii==3 or ii==11) and (jj==3 or jj==11)))
                    printf(" + ");
                else
                    printf(" - ");
            }
            else if(board[ii][jj][0]==1.f and board[ii][jj][1]==0.f)
                printf("\033[96m O \033[0m");
            else if(board[ii][jj][0]==0.f and board[ii][jj][1]==1.f)
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

    void set_state(float prior,int sidem);
    bool expanded();
    float value();
};
void NODE::set_state(float pr,int sidem)
{
    visit_count=0;
    to_play=sidem;
    prior=pr;
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
    int hash_size,table[2*maxPossibleMoves],initkey;
    float *values,*boardcheck;
    float* getValue(float *boardX, int *key_out);
    void setValue(int key_in, float *rst_in);
};

ZOBRIST::ZOBRIST(const int sze)
{
    auto lg=1;
    while(sze>>lg) lg++;
    hash_size=1<<(lg-1);
    printf("Hash size = %d\n",hash_size);
    for(auto i=0;i<2*maxPossibleMoves;i++)
        table[i]=mt_19937()%hash_size;
    initkey=mt_19937()%hash_size;
    values=new float[(maxPossibleMoves+1)*hash_size];
    boardcheck=new float[2*maxPossibleMoves*hash_size];
    for(auto i=0;i<2*maxPossibleMoves*hash_size;i++)
    boardcheck[i]=-1;
}
float* ZOBRIST::getValue(float *boardX, int *key_out)
{
    int key=initkey;
    for(auto ii=0;ii<2*maxPossibleMoves;ii++)
    {
        if(boardX[ii]==1.f) key^=table[ii];
    }
    *key_out=key;
    // printf("key: %d\n",key);
    if(memcmp(boardX,&boardcheck[key*2*maxPossibleMoves],sizeof(board)) == 0)
        return &values[key*(maxPossibleMoves+1)];
    else
        return NULL;

}

void ZOBRIST::setValue(int key_in, float *rst_in)
{
    memcpy(&values[key_in*(maxPossibleMoves+1)],rst_in,(maxPossibleMoves+1)*sizeof(float));
}

class A0ENGINE
{
public:
    TF_Session* Session;
    TF_Output* Input;
    TF_Output* Output;
    TF_Tensor** InputValues;
    TF_Tensor** OutputValues;
    int NumInputs = 1;
    int NumOutputs = 1;
    TF_Status* Status;
    float* engBoard;
    int initEngine(const char* saved_model_dir);
    void runEngine();
    ZOBRIST *zobrist;
    float eboard[15][15][2];
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
    Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0;
    
    //********* Get Output tensor
    Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    int ndims = 4;
    int64_t dims[] = {1,15,15,2};
    int ndata = sizeof(float)*450 ;// This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, eboard[0][0], ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL)
    {
        printf("TF_NewTensor is OK\n");
    }
    else
        printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;
    engBoard=(float*)TF_TensorData(InputValues[0]);

    zobrist=new ZOBRIST(60000);
    return 0;
}

void A0ENGINE::runEngine()
{
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0,NULL , Status);
    // if(TF_GetCode(Status) != TF_OK) printf("%s",TF_Message(Status));
}

float evaluate(NODE* node,A0ENGINE* engine)
{
    auto wld=winLossDraw();
    if(wld!=-1.f) return wld;
    float* policy_logits=NULL;
    int hskey;
    // policy_logits=NULL;
    engine->zobrist->getValue(board[0][0],&hskey);
    if(!policy_logits)//not found in hash table
    {
        // printf("not found!");
        if(side2move) memcpy(engine->engBoard,inv_board[0][0],sizeof(inv_board));
        else memcpy(engine->engBoard,board[0][0],sizeof(board));
        engine->runEngine();
        policy_logits=(float*)TF_TensorData(engine->OutputValues[0]);
        engine->zobrist->setValue(hskey,policy_logits);
    }
    auto nump=0;
    auto px=0,py=0;
    float policy[maxPossibleMoves],policy_sum=0;
    for(auto pos=0;pos<maxPossibleMoves;pos++)
    {
        px=pos/15;py=pos%15;
        if(board[px][py][0]==board[px][py][1])//since can not be both 1;
        {
            node->actions[nump]=pos;
            policy[nump]=policy_logits[pos];
            policy_sum += policy[nump];
            nump+=1;
        }
    }
    auto tp=1-node->to_play;
    for(auto idpolicy=0;idpolicy<nump;idpolicy++)
    {
        node->children[idpolicy]= &nodes[nodes_used];
        node->children[idpolicy]->set_state(policy[idpolicy]/policy_sum,tp);
        nodes_used++;
    }
    node->num_child=nump;
    float rvalue=side2move? 1-policy_logits[maxPossibleMoves] : policy_logits[maxPossibleMoves];
    TF_DeleteTensor(engine->OutputValues[0]);
    return rvalue;
}

std::gamma_distribution<float> GammaDistribution(.05,1.0);
std::uniform_real_distribution<float> UniformDistribution(0.0,1.0);

void add_exploration_noise(NODE *node)
{
    float noise[maxPossibleMoves],noise_sum=0;
    for(auto i=0;i<node->num_child;i++)
    {
        noise[i]=GammaDistribution(mt_19937);
        noise_sum+=noise[i];
    }
    for(auto i=0;i<node->num_child;i++)
    {
        node->children[i]->prior = node->children[i]->prior*.75+noise[i]/noise_sum*.25;
    }
}

float fpuReduction=1.3f,fpuReductionRoot=1.0f;
float pb_c;

float ucb_score(NODE *parent, NODE *child, bool isnotroot=true)
{
    float prior_score=pb_c/((float)(child->visit_count+1))*child->prior;
    if(child->visit_count==0)
        return isnotroot? (prior_score-fpuReduction+parent->value()) : (prior_score-fpuReductionRoot+parent->value());
    else
        return prior_score-child->value();
}

int select_child(NODE *node,bool isnotroot=true)
{
    // return mt_19937()%node->num_child;
    float maxscore=-1e10,score;
    int maxidx=0;
    pb_c=(logf((node->visit_count+19653)/19652.0f) + 1.25f) * sqrtf(node->visit_count);
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

void applyMove(int pos)
{
    auto px=pos/15,py=pos%15;
    board[px][py][side2move]=inv_board[px][py][1-side2move]=1;
    side2move=1-side2move;
    movelist[move_count]=pos;
    move_count+=1;
}
inline void applyMove(int px,int py){applyMove(px*15+py);}

void takeBack()
{
    move_count-=1;
    auto pos=movelist[move_count];
    auto px=pos/15,py=pos%15;
    side2move=1-side2move;
    board[px][py][side2move]=inv_board[px][py][1-side2move]=0;
}

void backpropagate(NODE **search_path,int lenth,float value)
{
    for(auto i=0;i<lenth;i++)
    {
        search_path[i]->value_sum += search_path[i]->to_play == 0? value : (1 - value);
        search_path[i]->visit_count += 1;
    }
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
    float maxscore=-1e10,cscore;
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
            cvcounts[i+1]=cvcounts[i]+pow(root->children[i]->visit_count/maxscore,1.5);
        }
        rnd=UniformDistribution(mt_19937)*cvcounts[root->num_child];
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
    rootnode.set_state(1.0,side2move);
    rootnode.visit_count=1;
    rootnode.value_sum=side2move==0? evaluate(&rootnode,engine) : 1.0f-evaluate(&rootnode,engine);
    // evaluate returns absolute value: 1 for black wins 0 for white wins
    int depth=0,maxdepth=0;
    if(dir_noise) add_exploration_noise(&rootnode);
    // printf("noise!\n");
    int idx;
    float value;
    NODE *search_path[maxPossibleMoves],*cr_node;
    for(auto _t=0;_t<num_simul;_t++)
    {
        cr_node=&rootnode;
        search_path[0]=cr_node;
        depth=1;

        while(cr_node->expanded())
        {
            idx=select_child(cr_node,depth>1);
            applyMove(cr_node->actions[idx]);
            cr_node=cr_node->children[idx];
            search_path[depth++]=cr_node;
        }
        // printBoard();
        value=evaluate(cr_node,engine);
        backpropagate(search_path,depth,value);
        for(auto tt=1;tt<depth;tt++) takeBack();        
        maxdepth=maxdepth>depth? maxdepth : depth;
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

void clearAll()
{
    move_count=0;
    side2move=0;
    for(auto px=0;px<15;px++)
        for(auto py=0;py<15;py++)
            board[px][py][0]=inv_board[px][py][0]=board[px][py][1]=inv_board[px][py][1]=0;
}

void hmpl()
{
    A0ENGINE a0eng;
    a0eng.initEngine("RNG_Old/");
    fpuReductionRoot=1.3f;
    float val,tmu;
    int act,dpt;
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
            if(0<=px and px<15 and 0<=py and py<15 and board[px][py][0]==board[px][py][0]) applyMove(px,py);
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
        case 'n':
            if(scanf("%d",&px)!=1) printf("Invalid command\n");
            else setNum_sml(px);
            break;
        case 'f':
            if(scanf("%f %f",&fpuReduction,&fpuReductionRoot)!=2) printf("Invalid command\n");
            printf("FPU = %f | %f\n",fpuReduction,fpuReductionRoot);
            break;
        case 'w':
            if(scanf("%f",&valueWt)!=1) printf("Invalid command\n");
            printf("valueWt = %f | %f\n",valueWt,valueWt*num_simul);
            valueWt=valueWt*num_simul;
            break;
        case 'b':
            if(move_count==0) printf("Cannot take back\n");
            else{printf("Take back\n");takeBack();}
            break;
        case 'c':
            printf("New game\n");
            clearAll();
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
    FILE *fpx,*fpy;
    sprintf(ff,"games/%s.x",out_file);
    fpx=fopen(ff,"a+b");
    if(!fpx){printf("error! %s\n",ff);return;}
    sprintf(ff,"games/%s.y",out_file);
    fpy=fopen(ff,"a+b");
    if(!fpy){printf("error! %s\n",ff);return;}
    while(totps<npos)
    {
        printf("game %d\n",ngames+1);
        clearAll();
        while(move_count==0 or (game_rst=winLossDraw())< -.5)
        {
            if(side2move) memcpy(&bufx[move_count*(maxPossibleMoves*2)],inv_board[0][0],sizeof(inv_board));
            else memcpy(&bufx[move_count*(maxPossibleMoves*2)],board[0][0],sizeof(board));
            // printf("copy complete\n");
            act=run_mcts(&a0eng,true,true,&bufy[move_count*(maxPossibleMoves+1)],&val,&dpt,&tmu);
            printf("\r%3d %2d %2d %.3f %3d %.3f ",move_count,act/15,act%15,val,dpt,tmu);
            fflush(stdout);
            applyMove(act);
        }
        for(auto kk=0;kk<move_count;kk++)
        {
            bufy[kk*(maxPossibleMoves+1)+maxPossibleMoves]=(move_count-kk)%2? game_rst : 1-game_rst;
        }
        ngames++;
        gs_sum+=game_rst;
        printf("\r");
        printBoard();
        totps+=move_count;
        printf("result: %.1f, %d games, %d pos, avg_score_black: %.3f\n",game_rst,ngames,totps,gs_sum/ngames);
        fwrite(bufx,sizeof(float),move_count*(maxPossibleMoves*2),fpx);
        fwrite(bufy,sizeof(float),move_count*(maxPossibleMoves+1),fpy);
    }
    printf("generated %d pos\n",totps);
    fclose(fpx);
    fclose(fpy);
    delete[] bufx;
    delete[] bufy;
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
    fpuReductionRoot=1.3f;
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
    printf("x-shift: %2d\n",(act=mt_19937()%3-1));
    for(auto kk=0;kk<nops;kk++) {opns[kk][0]+=act;}
    printf("y-shift: %2d\n",(act=mt_19937()%3-1));
    for(auto kk=0;kk<nops;kk++) {opns[kk][1]+=act;}

    sprintf(ff,"eval/%s.txt",out_file);
    fp=fopen(ff,"w");
    if(!fp){printf("error! %s\n",ff);return;}
    for(ngames=0;ngames<2;ngames++)
    {
        printf("game %d\n",ngames+1);
        clearAll();
        for(auto kk=0;kk<nops;kk++) 
        {
            fprintf(fp,"%3d b %2d %2d\n",move_count,opns[kk][0],opns[kk][1]);
            applyMove(opns[kk][0],opns[kk][1]);
        }
        while(move_count==0 or (game_rst=winLossDraw())< -.5)
        {
            side_id=(ngames+move_count)%2;
            if(side_id==0) act=run_mcts(&a0eng1,false,false,NULL,&val,&dpt,&tmu);
            else act=run_mcts(&a0eng2,false,false,NULL,&val,&dpt,&tmu);
            printf("\r%3d %d %2d %2d %.3f %3d %.3f ",move_count,side_id,act/15,act%15,val,dpt,tmu);
            fflush(stdout);
            fprintf(fp,"%3d %d %2d %2d %.3f %3d %.3f\n",move_count,side_id,act/15,act%15,val,dpt,tmu);
            applyMove(act);
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