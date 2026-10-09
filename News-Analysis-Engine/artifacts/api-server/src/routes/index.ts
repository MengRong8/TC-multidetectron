import { Router, type IRouter } from "express";
import healthRouter from "./health";
import detectionRouter from "./detection";

const router: IRouter = Router();

router.use(healthRouter);
router.use(detectionRouter);

export default router;
